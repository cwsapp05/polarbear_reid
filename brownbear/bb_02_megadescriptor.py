"""Brown bear stage 2: MegaDescriptor-L-384 embeddings of every sampled head crop (Windows, GPU).

Same transform as the PolarBearVidID run (resize to 384x384, normalise to [-1, 1]); the crop is the
dataset's own head crop, unpadded.

Output: work/features/megadescriptor.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import base_argparser, get_logger, load_config, read_csv  # noqa: E402

log = get_logger("bb.md")
HERE = Path(__file__).resolve().parent


class Crops:
    """Module-level so Windows DataLoader workers can pickle it."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def main() -> None:
    import timm
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work, m = cfg["paths"]["work_dir"], cfg["megadescriptor"]
    smp = read_csv(work / "sample.csv", "sample.csv (bb_00)")
    tf = T.Compose([T.Resize((m["image_size"], m["image_size"])), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(m["model"], num_classes=0, pretrained=True).eval().to(device)
    loader = DataLoader(Crops(smp["path"].tolist(), tf), batch_size=m["batch_size"], num_workers=m["num_workers"],
                        pin_memory=device == "cuda")
    feats = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                enabled=bool(m["fp16"]) and device == "cuda"):
        for x in tqdm(loader, desc="MegaDescriptor"):
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    (work / "features").mkdir(exist_ok=True)
    out = work / "features" / "megadescriptor.npz"
    np.savez_compressed(out, image_ids=np.asarray(smp["image_id"], dtype=object), feats=np.concatenate(feats))
    log.info("Saved %d MegaDescriptor embeddings -> %s", len(smp), out)


if __name__ == "__main__":
    main()
