"""Stage 3: extract re-ID features with MegaDescriptor (whole-body crop) and BearID (face chip).

MegaDescriptor runs on every MegaDetector crop, so it can also be scored on the full dataset.
BearID runs on every frame with a valid face (stage 2); subsets by face size are formed in stage 4.

Output: work/features/megadescriptor.npz, work/features/bearid.npz  (keys: image_ids, feats)
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import bearid_utils as bu
from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("features")


def save(path, image_ids, feats):
    np.savez_compressed(path, image_ids=np.asarray(image_ids, dtype=object), feats=np.asarray(feats, np.float32))
    log.info("Saved %d x %d features -> %s", len(image_ids), np.asarray(feats).shape[-1], path)


# --------------------------------------------------------------------------- MegaDescriptor
class Crops:
    """Map-style dataset over crop files. Module-level so DataLoader workers can pickle it on
    Windows, where worker processes are spawned rather than forked."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def megadescriptor(cfg, wp):
    import timm
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    mcfg = cfg["megadescriptor"]
    md = read_csv(wp["md_csv"], "megadetector.csv (stage 01)")
    md = md[md["md_ok"].astype(bool)].reset_index(drop=True)

    # transform from the MegaDescriptor model card
    tf = T.Compose([T.Resize((mcfg["image_size"], mcfg["image_size"])), T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No CUDA device found - MegaDescriptor-L-384 on CPU will be very slow.")
    model = timm.create_model(mcfg["model"], num_classes=0, pretrained=True).eval().to(device)
    loader = DataLoader(Crops(md["crop_path"].tolist(), tf), batch_size=mcfg["batch_size"], num_workers=mcfg["num_workers"],
                        pin_memory=device == "cuda")
    feats = []
    use_amp = bool(mcfg.get("fp16")) and device == "cuda"
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        for x in tqdm(loader, desc="MegaDescriptor"):
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    save(wp["features"] / "megadescriptor.npz", md["image_id"], np.concatenate(feats))


# --------------------------------------------------------------------------- BearID
def bearid(cfg, wp):
    faces = read_csv(wp["faces_csv"], "faces.csv (stage 02)")
    key_to_id = dict(zip(faces["bearid_input"], faces["image_id"]))
    bdir = wp["bearid_dir"]
    faces_xml = bdir / "faces_best.xml"
    if not faces_xml.exists():
        raise FileNotFoundError(f"{faces_xml} missing - run 02_face_filter.py first.")

    # bearchip: align each face on its eyes + nose and cut a 150x150 chip
    chip_dir = bdir / "chips"
    if chip_dir.exists():
        shutil.rmtree(chip_dir)
    bu.run([bu.binary(cfg, "bearchip"), "--root", str(wp["bearid_input"]) + "/", "--chipdir", chip_dir, faces_xml],
           cwd=bdir, log_path=bdir / "bearchip.log")
    chips_xml = bdir / "faces_best_chips.xml"
    chip_src = bu.parse_chips(chips_xml)
    log.info("bearchip produced %d chips", len(chip_src))

    # bearembed: 128-d embedding per chip
    emb_dir = bdir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    bu.run([bu.binary(cfg, "bearembed"), "--embed", bu.network(cfg, "embed_network"),
            "--output", emb_dir, chips_xml], cwd=bdir, log_path=bdir / "bearembed.log")
    embs = bu.parse_embeddings(bdir / "faces_best_chips_embeds.xml")

    ids, feats, missing = [], [], 0
    for chip, vec in embs.items():
        image_id = key_to_id.get(chip_src.get(chip, ""))
        if image_id is None:
            missing += 1
            continue
        ids.append(image_id)
        feats.append(vec)
    if missing:
        log.warning("%d embeddings could not be mapped back to a frame (check %s)", missing, chips_xml)
    if len(set(ids)) != len(ids):
        raise RuntimeError("Several chips map to the same frame; faces_best.xml should hold one face per frame.")
    save(wp["features"] / "bearid.npz", ids, np.stack(feats))


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--model", choices=["megadescriptor", "bearid", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp = work_paths(cfg)
    if args.model in ("bearid", "all"):
        bearid(cfg, wp)
    if args.model in ("megadescriptor", "all"):
        megadescriptor(cfg, wp)


if __name__ == "__main__":
    main()
