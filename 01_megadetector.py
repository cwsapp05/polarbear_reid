"""Stage 1: run MegaDetector on the sampled frames and save one animal crop per frame.

PolarBearVidID frames are already crops around the bear, so this stage mainly tightens the box
and documents how often MegaDetector agrees with the dataset's own crop. Frames with no animal
detection fall back to the full frame (configurable) and are flagged.

Output: work/megadetector.csv, work/crops/<id>/<frame>.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("megadetector")
ANIMAL = "1"  # MegaDetector category id for "animal"


def padded_box(bbox_norm, w, h, pad):
    """MegaDetector bbox [x_min, y_min, width, height] (normalised) -> padded pixel box."""
    x, y, bw, bh = bbox_norm
    x0, y0, x1, y1 = x * w, y * h, (x + bw) * w, (y + bh) * h
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(round(x0 - px))), max(0, int(round(y0 - py))),
            min(w, int(round(x1 + px))), min(h, int(round(y1 + py))))


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: process a random N frames spread across all bears")
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp = work_paths(cfg)
    mcfg = cfg["megadetector"]

    from megadetector.detection.run_detector import load_detector  # heavy import, keep local
    detector = load_detector(mcfg["model"], force_cpu=bool(mcfg.get("force_cpu", False)))

    sample = read_csv(wp["sample"], "sample.csv (stage 00)")
    if args.limit:
        sample = sample.sample(n=min(args.limit, len(sample)), random_state=0).sort_values("image_id")

    done = {}
    if wp["md_csv"].exists():  # resume support
        prev = pd.read_csv(wp["md_csv"])
        done = {r.image_id: r._asdict() for r in prev.itertuples(index=False)}
        log.info("Resuming: %d frames already processed", len(done))

    rows = []
    for i, r in enumerate(tqdm(sample.itertuples(index=False), total=len(sample), desc="MegaDetector")):
        crop_path = wp["crops"] / f"{r.image_id}.png"
        if r.image_id in done and crop_path.exists():
            rows.append(done[r.image_id])
            continue
        img = Image.open(r.path).convert("RGB")
        w, h = img.size
        res = detector.generate_detections_one_image(img, image_id=r.path,
                                                     detection_threshold=mcfg["conf_threshold"])
        dets = [d for d in (res.get("detections") or []) if d["category"] == ANIMAL]
        dets.sort(key=lambda d: d["conf"], reverse=True)

        if dets:
            box = padded_box(dets[0]["bbox"], w, h, mcfg["pad_frac"])
            conf, fallback = float(dets[0]["conf"]), False
        elif mcfg["fallback_to_full_image"]:
            box, conf, fallback = (0, 0, w, h), np.nan, True
        else:
            rows.append({"image_id": r.image_id, "md_ok": False})
            continue

        crop_path.parent.mkdir(parents=True, exist_ok=True)
        img.crop(box).save(crop_path)
        x0, y0, x1, y1 = box
        rows.append({
            "image_id": r.image_id, "md_ok": True, "md_fallback": fallback, "md_conf": conf,
            "md_n_animals": len(dets), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "img_w": w, "img_h": h, "box_area_frac": (x1 - x0) * (y1 - y0) / (w * h),
            "crop_path": str(crop_path),
        })
        if (i + 1) % 2000 == 0:
            pd.DataFrame(rows).to_csv(wp["md_csv"], index=False)

    out = pd.DataFrame(rows)
    out.to_csv(wp["md_csv"], index=False)

    md_ok = out["md_ok"].astype(bool)
    ok = out[md_ok]
    fb = ok["md_fallback"].astype(bool)
    log.info("Frames: %d | detection found: %d (%.1f%%) | fell back to full frame: %d | dropped: %d",
             len(out), (~fb).sum(), 100 * (~fb).mean(), fb.sum(), (~md_ok).sum())
    log.info("Median box area as fraction of the dataset crop: %.2f (1.0 = MegaDetector kept the whole crop)",
             ok.loc[~fb, "box_area_frac"].median())
    log.info("Frames with >1 animal detection: %d", (ok["md_n_animals"] > 1).sum())


if __name__ == "__main__":
    main()
