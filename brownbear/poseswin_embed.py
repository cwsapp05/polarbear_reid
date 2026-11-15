"""PoseSwin embeddings with the Mathis group's own code and released checkpoints.

Run with tools/poseswin-venv (torch from the main install + yacs, pytorch_metric_learning, albumentations).
Mirrors their inference path (tools/inference.py + processor.extract_feature): 224x224 resize, ImageNet
normalisation, feature = sum of original and horizontally flipped image, L2-normalised. Also records the
mean pose-keypoint confidence, which their pipeline uses to drop images (< 0.8).

  python poseswin_embed.py --year 2022 --images work/sample.csv --out work/features/poseswin_2022.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "tools" / "BrownBear_ReID" / "PoseGuidedReID"
sys.path.insert(0, str(REPO))
from project.config import cfg  # noqa: E402
from project.models.build_swint import build_swin_reid  # noqa: E402
from project.utils.tools import fuse_all_conv_bn, load_model  # noqa: E402

DATA = Path("D:/Temporary Downloads Nov 2026/Public_release/Public_release")


def build(year: int):
    c = cfg.clone()
    c.merge_from_file(str(REPO / "configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml"))
    c.merge_from_list(["MODEL.AGG_POSE_FEATURE", True, "MODEL.DIST_TRAIN", False, "MODEL.DEVICE_ID", "('0',)",
                       "MODEL.POSE_WEIGHT", str(DATA / "checkpoints/preprocessing_ckpts/pose/hrnet_w48_balanced_n13_refined.pth"),
                       "MODEL.PRETRAIN_PATH", "",
                       # the released config sets SWIN.EMBED_DIM=512, which their code uses for BOTH the backbone
                       # width and the re-ID embedding; the checkpoints are Swin-B (width 128) with a 512-d head
                       "MODEL.SWIN.EMBED_DIM", 128])
    c.freeze()
    log = logging.getLogger("poseswin")
    model = build_swin_reid(98, log, linear_num=512, cfg=c, load_weights=False, return_feature=True, device="cuda")
    w = DATA / f"checkpoints/reid_ckpts/2017_2022_six_exps/test_on_{year}/net_60.pth"
    model = load_model(model, str(w), logger=log, remove_fc=True)
    model = fuse_all_conv_bn(model.eval()).cuda()
    tf = T.Compose([T.Resize(list(c.INPUT.IMG_SIZE)), T.ToTensor(), T.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD)])
    return model, tf


@torch.no_grad()
def embed(model, tf, paths, batch=32):
    feats, confs = [], []
    for k in tqdm(range(0, len(paths), batch), desc="PoseSwin"):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[k:k + batch]]).cuda()
        ff, cf = 0, []
        for flip in (False, True):
            out = model(torch.flip(x, dims=[3]) if flip else x)
            ff = ff + out[1]
            pc = out[2].detach().cpu().numpy() if torch.is_tensor(out[2]) else np.asarray(out[2])
            cf.append(pc.reshape(len(x), -1).mean(1))
        feats.append(torch.nn.functional.normalize(ff, dim=1).cpu().numpy())
        confs.append(cf[0])          # their filter uses the un-flipped pass's confidence first
    return np.concatenate(feats), np.concatenate(confs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--images", required=True, help="csv with image_id and path columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.images)
    if a.limit:
        df = df.head(a.limit)
    model, tf = build(a.year)
    f, c = embed(model, tf, df["path"].tolist())
    np.savez_compressed(a.out, image_ids=np.asarray(df["image_id"], dtype=object), feats=f, pose_conf=c)
    print(f"saved {len(df)} -> {a.out}; pose_conf median {np.median(c):.3f}, <0.8: {(c < 0.8).mean():.1%}")


if __name__ == "__main__":
    main()
