"""Shared MegaDescriptor embedding helper (importable module, so Windows DataLoader workers can pickle Crops)."""
from __future__ import annotations

import re

import numpy as np
from PIL import Image
from tqdm import tqdm

_WSL = re.compile(r"^/mnt/([a-z])/")


def win_path(p: str) -> str:
    """'/mnt/c/a/b' -> 'C:/a/b' (BearID XML written under WSL holds WSL paths)."""
    m = _WSL.match(p)
    return f"{m.group(1).upper()}:/" + p[m.end():] if m else p


class Crops:
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def embed(paths, mcfg: dict, desc="MegaDescriptor") -> np.ndarray:
    import timm
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    tf = T.Compose([T.Resize((mcfg["image_size"], mcfg["image_size"])), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(mcfg["model"], num_classes=0, pretrained=True).eval().to(device)
    loader = DataLoader(Crops(paths, tf), batch_size=mcfg["batch_size"], num_workers=mcfg["num_workers"],
                        pin_memory=device == "cuda")
    out = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                enabled=bool(mcfg.get("fp16")) and device == "cuda"):
        for x in tqdm(loader, desc=desc):
            out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out)


def save(path, image_ids, feats):
    np.savez_compressed(path, image_ids=np.asarray(image_ids, dtype=object), feats=np.asarray(feats, np.float32))
