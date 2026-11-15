"""Stage 9: same model, different crop. Is it the input (body vs face) or the model?

MegaDescriptor embeds three inputs for every face-visible frame of the primary subset:
  body      the MegaDetector body crop (stage 3, unchanged)
  facebox   BearID's detected face box, cut from the body crop with a 15% margin
  chip      BearID's aligned 150x150 face chip (the exact input BearID embeds)
and is scored next to BearID on exactly the stage-4 frames and pairs.

  MD facebox/chip - MD body   -> input effect, model held fixed
  BearID - MD chip            -> model effect, input held fixed

Designed after the stage-8 results; secondary analysis.
Output: work/faces_box/, work/features/megadescriptor_{facebox,chip}.npz, results/same_model.md, .json
"""
from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import bearid_utils as bu
import md_embed
from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("same_model")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("evaluate", "04_evaluate.py")
E7 = _load("masked_eval", "07_masked_eval.py")

MD, FB, CH, BD = "megadescriptor", "megadescriptor_facebox", "megadescriptor_chip", "bearid"
NAMES = {MD: "MD, body crop", FB: "MD, face box", CH: "MD, aligned face chip", BD: "BearID (face chip)",
         "fused": "Fused (MD body + BearID)"}
E7.DELTAS = [(FB, MD), (CH, MD), (CH, FB), (BD, CH), (BD, FB), (BD, MD)]
E7.SIM.update({FB: "cosine", CH: "cosine"})
MARGIN = 0.15


def face_boxes(wp, faces, md):
    """faces_best.xml boxes (upscaled-input coords) -> face crops cut from the body crop."""
    key_to_id = dict(zip(faces["bearid_input"], faces["image_id"]))
    crop = md.set_index("image_id")["crop_path"]
    out_dir = wp["work"] / "faces_box"
    rows = []
    for img in ET.parse(wp["bearid_dir"] / "faces_best.xml").getroot().iter("image"):
        iid = key_to_id.get(img.get("file"))
        box = img.find("box")
        if iid is None or box is None:
            continue
        body = Image.open(crop[iid]).convert("RGB")
        up = Image.open(md_embed.win_path(img.get("file")))
        s = max(up.size) / max(body.size)                       # upscale factor used in stage 2
        l, t, w, h = (float(box.get(k)) / s for k in ("left", "top", "width", "height"))
        mx, my = MARGIN * w, MARGIN * h
        x0, y0 = max(0, round(l - mx)), max(0, round(t - my))
        x1, y1 = min(body.width, round(l + w + mx)), min(body.height, round(t + h + my))
        p = out_dir / f"{iid}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        body.crop((x0, y0, x1, y1)).save(p)
        rows.append((iid, str(p), x1 - x0))
    return pd.DataFrame(rows, columns=["image_id", "path", "w_px"])


def chips(wp, faces):
    key_to_id = dict(zip(faces["bearid_input"], faces["image_id"]))
    src = bu.parse_chips(wp["bearid_dir"] / "faces_best_chips.xml")
    rows = [(key_to_id[s], md_embed.win_path(c)) for c, s in src.items() if s in key_to_id]
    return pd.DataFrame(rows, columns=["image_id", "path"])


def main() -> None:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    wp, ecfg, bcfg = work_paths(cfg), cfg["evaluation"], cfg["bearid"]
    wp["work"] = cfg["paths"]["work_dir"]
    reps, seed, k = int(ecfg["bootstrap_reps"]), int(ecfg["seed"]), int(ecfg["max_frames_per_sequence"])
    res_dir = cfg["paths"]["results_dir"]

    faces = pd.read_csv(wp["faces_csv"])           # keep WSL keys as written (match the XML)
    md = read_csv(wp["md_csv"], "megadetector.csv")
    for name, table in ((FB, lambda: face_boxes(wp, faces, md)), (CH, lambda: chips(wp, faces))):
        out = wp["features"] / f"{name}.npz"
        if not out.exists():
            t = table()
            log.info("%s: %d inputs", name, len(t))
            md_embed.save(out, t["image_id"], md_embed.embed(t["path"], cfg["megadescriptor"], name))

    # identical frame selection to stages 4 and 7
    sample = read_csv(wp["sample"], "sample.csv")
    feats = {m: E.load_feats(wp["features"] / f"{m}.npz") for m in (MD, FB, CH, BD)}
    df = sample.merge(faces.drop(columns="bearid_input"), on="image_id", how="left")
    df["zoo"] = df["zoo"].fillna("unknown").astype(str)
    df["identity"] = df["identity"].astype(str)
    has = {m: df["image_id"].isin(feats[m].keys()) for m in feats}
    w = int(bcfg["primary_min_face_width_px"])
    mask = df["landmarks_ok"].fillna(False).astype(bool) & (df["face_w_px"] >= w) & has[BD] & has[MD]
    sub, _ = E.select_frames(df[mask], k, seed)
    missing = {m: int((~sub["image_id"].isin(feats[m].keys())).sum()) for m in (FB, CH)}
    assert not any(missing.values()), f"missing face inputs: {missing}"
    S = E7.scores_for(sub, (MD, FB, CH, BD), feats)
    S["fused"] = (E.zscore_pairs(S[MD]) + E.zscore_pairs(S[BD])) / 2
    out = {"primary": E7.evaluate(f"face_w>={w}px", sub, S, reps, seed)}
    (res_dir / "same_model.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    E7.NAMES = NAMES
    out["mask_stats"] = {"n": 0, "frac_median": 0, "flagged": 0}
    write_md(out, res_dir / "same_model.md", reps)
    log.info("Done. See %s", res_dir / "same_model.md")


def write_md(out, path, reps):
    r = out["primary"]
    L = ["# Same model, different crop (stage 9)", "",
         "MegaDescriptor on the body crop, on BearID's face box (15% margin) and on BearID's aligned chip, "
         "next to BearID, on the stage-4 primary frames and pairs. Designed after stage 8; secondary.", "",
         f"## {r['subset']} ({r['n_images']} images, {r['n_sequences']} sequences, {r['n_bears']} bears)", ""]
    models = list(r["negatives"]["all"]["auroc"])
    L += ["| Negatives | " + " | ".join(NAMES[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    for mode, n in r["negatives"].items():
        ci = n["bootstrap"]["sequence"]["ci95"]
        L.append(f"| {mode.replace('_', '-')} | " + " | ".join(
            f"{n['auroc'][m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]" for m in models) + " |")
    L += ["", "| Negatives | Comparison | Unit | Δ [95% CI] | p |", "|---|---|---|---|---|"]
    for mode, n in r["negatives"].items():
        for unit, b in n["bootstrap"].items():
            for kk, d in b["deltas"].items():
                a, c = kk.split("-")
                L.append(f"| {mode.replace('_', '-')} | {NAMES[a]} − {NAMES[c]} | {'bear' if unit == 'identity' else unit} | "
                         f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {E.fmt_p(d['p_bootstrap'], reps)} |")
    L += ["", "| Model | Rank-1 | mAP |", "|---|---|---|"]
    L += [f"| {NAMES[m]} | {v['rank1']:.3f} | {v['mAP']:.3f} |" for m, v in r["ranking"].items()]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
