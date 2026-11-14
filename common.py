"""Shared helpers: config, paths, logging and the PolarBearVidID catalogue."""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- config
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_DIR / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = path.resolve().parent
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else (base / p)
    return cfg


def base_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: next to the scripts)")
    return ap


def get_logger(name: str) -> logging.Logger:
    # own handler + no propagation: some libraries (e.g. megadetector) attach root handlers too
    logger = logging.getLogger(f"pbreid.{name}")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- work-dir layout
def work_paths(cfg: dict) -> dict[str, Path]:
    w = cfg["paths"]["work_dir"]
    d = {
        "catalogue": w / "catalogue.csv",            # every frame in the dataset
        "sample": w / "sample.csv",                  # frames kept after frame_stride
        "md_csv": w / "megadetector.csv",            # one row per sampled frame
        "crops": w / "crops",                        # MegaDetector crops (native resolution)
        "bearid_dir": w / "bearid",
        "bearid_input": w / "bearid" / "input",      # upscaled crops fed to bearface
        "faces_csv": w / "faces.csv",                # face-filter result per frame
        "features": w / "features",
    }
    for key in ("crops", "bearid_dir", "bearid_input", "features"):
        d[key].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["results_dir"].mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- dataset
def find_data_root(cfg: dict) -> Path:
    """Return the folder holding animal_db.csv (the zip may add an extra top-level folder)."""
    root = cfg["paths"]["data_root"]
    if (root / "animal_db.csv").exists():
        return root
    hits = sorted(cfg["paths"]["download_dir"].rglob("animal_db.csv"))
    if not hits:
        raise FileNotFoundError(
            f"animal_db.csv not found under {root} or {cfg['paths']['download_dir']}. "
            "Run 00_prepare_data.py first or fix paths.data_root in config.yaml.")
    return hits[0].parent


def sequence_of(fname: str, regex: str | None) -> int:
    if regex:
        m = re.search(regex, fname)
        if not m:
            raise ValueError(f"sequence_regex {regex!r} did not match file name {fname!r}")
        return int(m.group("seq"))
    return int(fname[7:10])  # convention used by the wildlife-datasets PolarBearVidID loader


def build_catalogue(cfg: dict) -> pd.DataFrame:
    root = find_data_root(cfg)
    meta = pd.read_csv(root / "animal_db.csv")
    meta.columns = [c.strip().lower() for c in meta.columns]
    regex = cfg["dataset"].get("sequence_regex")

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            rows.append({"path": str(f), "file": f.name, "id": int(d.name),
                         "seq": sequence_of(f.name, regex)})
    if not rows:
        raise RuntimeError(f"No images found in numbered identity folders under {root}")
    df = pd.DataFrame(rows)
    df = df.merge(meta, on="id", how="left")
    df["identity"] = df["name"].fillna(df["id"].astype(str)) if "name" in df else df["id"].astype(str)
    if "zoo" not in df:
        df["zoo"] = "unknown"
    df["sequence"] = df["id"].astype(str) + "_" + df["seq"].astype(str)  # unique across bears
    df["frame_idx"] = df.groupby("sequence").cumcount()
    df["image_id"] = df["id"].astype(str) + "/" + df["file"].str.rsplit(".", n=1).str[0]
    return df[["image_id", "path", "file", "id", "identity", "zoo", "sequence", "frame_idx"]]


_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/]")


def _win_to_wsl(p):
    """'C:\\a\\b' -> '/mnt/c/a/b' (only used when running on Linux, e.g. BearID stages under WSL)."""
    if isinstance(p, str):
        m = _WIN_PATH.match(p)
        if m:
            return f"/mnt/{m.group(1).lower()}/" + p[3:].replace("\\", "/")
    return p


def read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run the stage that produces {what} first.")
    df = pd.read_csv(path)
    if os.name == "posix":  # CSVs written by the Windows-side stages hold Windows paths
        for col in ("path", "crop_path", "bearid_input"):
            if col in df:
                df[col] = df[col].map(_win_to_wsl)
    return df
