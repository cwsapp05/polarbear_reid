"""Stage 2: decide which frames show the face well enough for face-based re-ID.

Runs BearID's face detector + landmark model (bearface) on each MegaDetector crop. A frame is
"face visible at width w" when a face is found, all six landmarks fall inside the face box, and
the face box is at least w pixels wide in ORIGINAL image pixels. Both models are later scored
on exactly these subsets.

Output: work/faces.csv, work/bearid/faces_best.xml (largest valid face per frame, for stage 3),
        results/face_width_hist.png, results/face_visibility_by_bear.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import bearid_utils as bu
from common import base_argparser, get_logger, load_config, read_csv, work_paths
from plotting import face_width_histogram

log = get_logger("face_filter")


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp = work_paths(cfg)
    bcfg = cfg["bearid"]

    md = read_csv(wp["md_csv"], "megadetector.csv (stage 01)")
    md = md[md["md_ok"].astype(bool)]
    sample = read_csv(wp["sample"], "sample.csv (stage 00)").set_index("image_id")

    # 1. upscale crops so tiny faces are detectable; remember the scale to convert sizes back
    target = int(bcfg["upscale_long_side"])
    scale, labels, inputs = {}, {}, []
    for r in tqdm(md.itertuples(index=False), total=len(md), desc="Upscaling crops"):
        dst = wp["bearid_input"] / f"{r.image_id}.png"
        img = Image.open(r.crop_path)
        s = target / max(img.size)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS).save(dst)
        key = str(dst)
        scale[key], labels[key] = s, str(sample.loc[r.image_id, "id"])
        inputs.append((r.image_id, key))

    # 2. bearface
    faces = bu.detect_faces(cfg, [k for _, k in inputs], wp["bearid_dir"] / "bearface",
                            n_workers=int(bcfg.get("n_workers", 1)), logger=log)

    # 3. per-frame face record (sizes in original pixels)
    rows, best = [], {}
    for image_id, key in inputs:
        found = faces.get(key, [])
        f, lm_ok, eye = bu.best_face(found, float(bcfg["landmark_margin_frac"]))
        s = scale[key]
        rows.append({
            "image_id": image_id, "n_faces": len(found),
            "face_w_px": f["width"] / s if f else np.nan,
            "face_h_px": f["height"] / s if f else np.nan,
            "landmarks_ok": bool(lm_ok), "eye_dist_frac": eye, "bearid_input": key,
        })
        if f is not None and lm_ok:
            best[key] = f
    out = pd.DataFrame(rows)
    out.to_csv(wp["faces_csv"], index=False)
    bu.write_faces(best, labels, wp["bearid_dir"] / "faces_best.xml")

    # 4. report
    valid = out["landmarks_ok"]
    log.info("Frames: %d | any face: %d (%.1f%%) | face with valid landmarks: %d (%.1f%%)",
             len(out), (out.n_faces > 0).sum(), 100 * (out.n_faces > 0).mean(), valid.sum(), 100 * valid.mean())
    widths = out.loc[valid, "face_w_px"]
    if len(widths):
        log.info("Face width (original px) of valid faces: median %.1f, IQR %.1f-%.1f",
                 widths.median(), widths.quantile(.25), widths.quantile(.75))
    thresholds = [int(w) for w in bcfg["min_face_widths_px"]]
    for w in thresholds:
        log.info("  face visible at >= %2d px: %d frames", w, (valid & (out.face_w_px >= w)).sum())

    # visibility per bear: if some bears rarely show their face, the face subset is not a random
    # sample of the population - worth reporting alongside the AUROCs
    per = out.merge(sample[["identity", "sequence"]], left_on="image_id", right_index=True)
    w0 = int(bcfg["primary_min_face_width_px"])
    per["visible"] = per["landmarks_ok"] & (per["face_w_px"] >= w0)
    tab = per.groupby("identity").agg(frames=("image_id", "size"), face_visible=("visible", "sum"),
                                      sequences_with_face=("sequence", lambda s: s[per.loc[s.index, "visible"]].nunique()))
    tab["face_visible_frac"] = (tab["face_visible"] / tab["frames"]).round(3)
    res = cfg["paths"]["results_dir"]
    tab.to_csv(res / "face_visibility_by_bear.csv")
    log.info("Face visibility by bear at >= %d px:\n%s", w0, tab.to_string())
    face_width_histogram(widths.to_numpy(), thresholds, w0, res / "face_width_hist.png")
    log.info("Wrote %s and face_width_hist.png. Pick primary_min_face_width_px from this histogram "
             "BEFORE looking at AUROCs.", wp["faces_csv"])


if __name__ == "__main__":
    main()
