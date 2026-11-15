"""Brown bear stage 4: MegaDescriptor on BearID's aligned face chips vs on the raw head crops.

Same frames and pair sets as bb_03 (crops where BearID found a face). Holds the model fixed and
varies the input: raw head crop -> aligned face chip. BearID - MD(chip) is then the model effect
on an identical input. Secondary analysis, designed after bb_03.

Output: work/features/megadescriptor_chip.npz, results/md_on_chips.md, .json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bearid_utils as bu  # noqa: E402
import md_embed  # noqa: E402
import metrics as M  # noqa: E402
from common import base_argparser, get_logger, load_config, read_csv  # noqa: E402

HERE = Path(__file__).resolve().parent
log = get_logger("bb.chips")
spec = importlib.util.spec_from_file_location("bb03", HERE / "bb_03_evaluate.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

MD, CH, BD = "megadescriptor", "megadescriptor_chip", "bearid"
B.NAMES = {MD: "MD, head crop", CH: "MD, aligned face chip", BD: "BearID (face chip)", "fused": "Fused"}
B.SIM[CH] = "cosine"
B.DELTAS = [(CH, MD), (BD, CH), (BD, MD)]


def main() -> None:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work, res = cfg["paths"]["work_dir"], cfg["paths"]["results_dir"]
    reps, seed = int(cfg["evaluation"]["bootstrap_reps"]), int(cfg["evaluation"]["seed"])
    smp = pd.read_csv(work / "sample.csv")
    norm = lambda p: str(Path(md_embed.win_path(p)))  # noqa: E731  (WSL or Windows path -> one form)
    key_to_id = {norm(k): i for k, i in zip(smp["bearid_input"], smp["image_id"])}
    out = work / "features" / f"{CH}.npz"
    if not out.exists():
        src = bu.parse_chips(work / "bearid" / "full" / "faces_best_chips.xml")
        rows = [(key_to_id[norm(s)], norm(c)) for c, s in src.items() if norm(s) in key_to_id]
        log.info("%d chips", len(rows))
        ids, paths = zip(*rows)
        md_embed.save(out, ids, md_embed.embed(paths, cfg["megadescriptor"], "MD on chips"))

    faces = read_csv(work / "faces.csv", "faces.csv")
    feats = {m: B.load_feats(work / "features" / f"{m}.npz") for m in (MD, CH, BD)}
    df = smp.merge(faces, on="image_id", how="left")
    sub = df[df["image_id"].isin(feats[BD])].reset_index(drop=True)
    n_enc = sub.groupby("identity")["encounter"].nunique()
    sub = sub[sub["identity"].isin(n_enc[n_enc >= 2].index)].reset_index(drop=True)
    assert sub["image_id"].isin(feats[CH]).all(), "chip features missing"
    S = {m: M.similarity(np.stack([feats[m][k] for k in sub["image_id"]]), B.SIM[m]) for m in (MD, CH, BD)}
    r = B.evaluate("bearid_ok", sub, S, reps, seed, ("encounter", "identity"))
    (res / "md_on_chips.json").write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
    B.write_md({"coverage": {"sampled": len(df), "bearid_embedded": len(sub)}, "bearid_ok": r,
                "all_frames": r}, res / "md_on_chips.md", reps)
    txt = (res / "md_on_chips.md").read_text(encoding="utf-8")   # bb_03's writer prints two subsets; keep one
    first = txt.index("\n## ")
    txt = txt[:txt.find("\n## ", first + 1)] if txt.find("\n## ", first + 1) > 0 else txt
    (res / "md_on_chips.md").write_text(txt.replace("Brown bears (McNeil River heads): MegaDescriptor vs BearID",
                                                    "Brown bears: MegaDescriptor on head crops vs aligned face chips"),
                                        encoding="utf-8")
    log.info("Done. See %s", res / "md_on_chips.md")


if __name__ == "__main__":
    main()
