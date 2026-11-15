"""Brown bear stage 1 (run in WSL): BearID face detection, chips and embeddings on the head crops.

bearface -> largest face with all six landmarks inside the box -> bearchip (aligned 150x150 chip)
-> bearembed (128-d). Frames where no valid face is found have no BearID embedding; stage 3
scores both models on the frames BearID could embed (and MegaDescriptor on all frames as well).

  python bb_01_bearid.py                  # uses work/sample.csv
  python bb_01_bearid.py --smoke          # uses work/sample_smoke.csv, detection only

Output: work/faces.csv, work/features/bearid.npz
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bearid_utils as bu  # noqa: E402
from common import base_argparser, get_logger, load_config, read_csv  # noqa: E402

log = get_logger("bb.bearid")
HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work, b = cfg["paths"]["work_dir"], cfg["bearid"]
    smp = read_csv(work / ("sample_smoke.csv" if args.smoke else "sample.csv"), "sample.csv (bb_00)")
    bdir = work / "bearid" / ("smoke" if args.smoke else "full")
    bdir.mkdir(parents=True, exist_ok=True)

    faces = bu.detect_faces(cfg, smp["bearid_input"].tolist(), bdir / "bearface",
                            n_workers=int(b["n_workers"]), logger=log)
    rows, best, labels = [], {}, {}
    for r in smp.itertuples(index=False):
        found = faces.get(r.bearid_input, [])
        f, ok, eye = bu.best_face(found, float(b["landmark_margin_frac"]), tuple(b["required_landmarks"]))
        # face box relative to the original head crop (fraction of its width)
        fw = f["width"] / r.bearid_scale / r.orig_w if f else np.nan
        rows.append({"image_id": r.image_id, "n_faces": len(found), "landmarks_ok": bool(ok),
                     "face_w_frac": fw, "eye_dist_frac": eye})
        if f is not None and ok:
            best[r.bearid_input] = f
            labels[r.bearid_input] = r.identity
    fdf = pd.DataFrame(rows)
    fdf.to_csv(bdir / "faces.csv" if args.smoke else work / "faces.csv", index=False)
    log.info("Images %d | any face %d (%.1f%%) | valid landmarks %d (%.1f%%) | face width / crop width: median %.2f",
             len(fdf), (fdf.n_faces > 0).sum(), 100 * (fdf.n_faces > 0).mean(), fdf.landmarks_ok.sum(),
             100 * fdf.landmarks_ok.mean(), fdf.loc[fdf.landmarks_ok, "face_w_frac"].median())
    if args.smoke:
        return

    faces_xml = bdir / "faces_best.xml"
    bu.write_faces(best, labels, faces_xml)
    chip_dir = bdir / "chips"
    if chip_dir.exists():
        shutil.rmtree(chip_dir)
    in_dir = Path(smp["bearid_input"].iloc[0]).parent
    bu.run([bu.binary(cfg, "bearchip"), "--root", str(in_dir) + "/", "--chipdir", chip_dir, faces_xml],
           cwd=bdir, log_path=bdir / "bearchip.log")
    chips_xml = bdir / "faces_best_chips.xml"
    chip_src = bu.parse_chips(chips_xml)
    log.info("bearchip produced %d chips", len(chip_src))
    emb_dir = bdir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    bu.run([bu.binary(cfg, "bearembed"), "--embed", bu.network(cfg, "embed_network"), "--output", emb_dir, chips_xml],
           cwd=bdir, log_path=bdir / "bearembed.log")
    embs = bu.parse_embeddings(bdir / "faces_best_chips_embeds.xml")
    key_to_id = dict(zip(smp["bearid_input"], smp["image_id"]))
    ids, feats = [], []
    for chip, vec in embs.items():
        iid = key_to_id.get(chip_src.get(chip, ""))
        if iid is not None:
            ids.append(iid); feats.append(vec)
    if len(set(ids)) != len(ids):
        raise RuntimeError("several chips map to one image")
    (work / "features").mkdir(exist_ok=True)
    np.savez_compressed(work / "features" / "bearid.npz", image_ids=np.asarray(ids, dtype=object), feats=np.stack(feats))
    log.info("Saved %d BearID embeddings (%d of %d images unmapped/no face)", len(ids), len(smp) - len(ids), len(smp))


if __name__ == "__main__":
    main()
