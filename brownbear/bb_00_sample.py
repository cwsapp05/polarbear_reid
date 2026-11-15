"""Brown bear stage 0: catalogue the head crops, sample frames per encounter, prepare BearID inputs.

Catalogue = union of every split CSV in data/reid_annotations (de-duplicated by image).
Encounter = (bear, calendar day). Up to `images_per_encounter` random photos per encounter are
kept; bears seen on only one day are dropped (no scorable positives).

For BearID, each crop is resized to `input_long_side` and padded with grey by `pad_frac` on every
side, because bearface looks for a face inside a scene and the crops are already tight heads.

  python bb_00_sample.py              # full sample
  python bb_00_sample.py --limit 100  # smoke test (writes sample_smoke.csv)

Output: work/catalogue.csv, work/sample.csv, work/bearid/input/*.png
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import base_argparser, get_logger, load_config  # noqa: E402

log = get_logger("bb.sample")
HERE = Path(__file__).resolve().parent


def catalogue(root: Path) -> pd.DataFrame:
    fs = sorted((root / "data" / "reid_annotations").glob("*/*.csv"))
    df = pd.concat([pd.read_csv(f, usecols=["image", "id", "year", "timestamp", "camera"]) for f in fs])
    df = df.drop_duplicates("image").reset_index(drop=True)
    df["path"] = [str(root / "images" / p) for p in df["image"]]
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    df["identity"] = df["id"].astype(str)
    df["encounter"] = df["identity"] + "@" + df["date"]
    df["image_id"] = df["image"].str.replace(r"\.[^.]+$", "", regex=True)
    return df[["image_id", "path", "identity", "year", "date", "encounter", "camera", "timestamp"]]


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work = cfg["paths"]["work_dir"]
    work.mkdir(parents=True, exist_ok=True)
    root = cfg["paths"]["data_root"]

    cat = catalogue(root)
    missing = ~cat["path"].map(lambda p: Path(p).exists())
    if missing.any():
        log.warning("%d catalogued images not found on disk; dropped", int(missing.sum()))
        cat = cat[~missing]
    cat.to_csv(work / "catalogue.csv", index=False)
    log.info("Catalogue: %d images, %d bears, %d encounters, years %s", len(cat), cat["identity"].nunique(),
             cat["encounter"].nunique(), sorted(cat["year"].unique()))

    s = cfg["sampling"]
    smp = cat.sample(frac=1.0, random_state=s["seed"]).groupby("encounter").head(int(s["images_per_encounter"]))
    n_enc = smp.groupby("identity")["encounter"].nunique()
    smp = smp[smp["identity"].isin(n_enc[n_enc >= 2].index)].sort_values("image_id").reset_index(drop=True)
    if args.limit:
        smp = smp.sample(n=args.limit, random_state=0).reset_index(drop=True)
    log.info("Sample: %d images, %d bears, %d encounters (dropped %d single-day bears)", len(smp),
             smp["identity"].nunique(), smp["encounter"].nunique(), int((n_enc < 2).sum()))

    b = cfg["bearid"]
    L, pad = int(b["input_long_side"]), float(b["pad_frac"])
    in_dir = work / "bearid" / "input"
    keys, scales, offs, sizes = [], [], [], []
    for r in tqdm(smp.itertuples(index=False), total=len(smp), desc="BearID inputs"):
        dst = in_dir / f"{hashlib.md5(r.image_id.encode()).hexdigest()}.png"  # ASCII-only: BearID XML is Latin-1
        img = Image.open(r.path)
        img = img.convert("RGB")
        sc = L / max(img.size)
        w, h = max(1, round(img.width * sc)), max(1, round(img.height * sc))
        px, py = round(pad * w), round(pad * h)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            canvas = Image.new("RGB", (w + 2 * px, h + 2 * py), (128, 128, 128))
            canvas.paste(img.resize((w, h), Image.LANCZOS), (px, py))
            canvas.save(dst)
        keys.append(str(dst)); scales.append(sc); offs.append((px, py)); sizes.append(img.size)
    smp["bearid_input"] = keys
    smp["bearid_scale"] = scales
    smp["pad_x"], smp["pad_y"] = zip(*offs)
    smp["orig_w"], smp["orig_h"] = zip(*sizes)
    out = work / ("sample_smoke.csv" if args.limit else "sample.csv")
    smp.to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
