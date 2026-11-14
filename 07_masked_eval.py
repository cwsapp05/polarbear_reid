"""Stage 7: does MegaDescriptor still beat BearID once the background is removed?

Scores MegaDescriptor on the SAM-masked crops (stage 6) next to the original MegaDescriptor,
BearID and both fusions, on exactly the frames and pairs stage 4 used (same selection, same seed):

  * primary face-visible subset (face width >= primary_min_face_width_px)
  * all frames (MegaDescriptor original vs masked only)

If MegaDescriptor's lead over BearID came from the enclosure, masking should cost it most on
ALL negatives (where cross-zoo context helps) and little on WITHIN-ZOO negatives.

Output: results/masked.md, results/masked.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import metrics as M
from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("masked")
_spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).with_name("04_evaluate.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

MD, MDM, BD = "megadescriptor", "megadescriptor_masked", "bearid"
NAMES = {MD: "MegaDescriptor", MDM: "MegaDescriptor, masked", BD: "BearID",
         "fused": "Fused (MD + BearID)", "fused_masked": "Fused (masked MD + BearID)"}
DELTAS = [(MDM, MD), (BD, MDM), (BD, MD), ("fused_masked", MDM), ("fused_masked", "fused")]
SIM = {MD: "cosine", MDM: "cosine", BD: "neg_euclidean"}


def scores_for(df, models, feats):
    return {m: M.similarity(np.stack([feats[m][i] for i in df["image_id"]]), SIM[m]) for m in models}


def evaluate(name, df, S, reps, seed, units=("sequence", "identity")):
    ident, seq, zoo = (df[c].to_numpy().astype(str) for c in ("identity", "sequence", "zoo"))
    out = {"subset": name, "n_images": len(df), "n_sequences": int(df["sequence"].nunique()),
           "n_bears": int(df["identity"].nunique()), "negatives": {}}
    for mode in ("all", "within_zoo"):
        i, j, y = M.make_pairs(ident, seq, zoo, within_zoo=(mode == "within_zoo"))
        sc = {m: s[i, j] for m, s in S.items()}
        r = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auroc": {m: float(roc_auc_score(y, s)) for m, s in sc.items()},
             "bootstrap": {}}
        for unit in units:
            codes, u = pd.factorize(df[unit])
            boot = M.paired_bootstrap(sc, y, codes[i], codes[j], len(u), reps, seed)
            b = {"ci95": {m: list(M.ci(v)) for m, v in boot.items()}, "deltas": {}}
            for a, c in DELTAS:
                if a in sc and c in sc:
                    d = boot[a] - boot[c]
                    b["deltas"][f"{a}-{c}"] = {"point": r["auroc"][a] - r["auroc"][c], "ci95": list(M.ci(d)),
                                               "p_bootstrap": M.bootstrap_p(d)}
            r["bootstrap"][unit] = b
        out["negatives"][mode] = r
        log.info("[%s | %s] %s", name, mode, "  ".join(f"{m}: {v:.3f}" for m, v in r["auroc"].items()))
    out["ranking"] = {m: M.rank_metrics(s, ident, seq) for m, s in S.items()}
    return out


def main() -> None:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    wp, ecfg, bcfg = work_paths(cfg), cfg["evaluation"], cfg["bearid"]
    reps, seed, k = int(ecfg["bootstrap_reps"]), int(ecfg["seed"]), int(ecfg["max_frames_per_sequence"])
    res_dir = cfg["paths"]["results_dir"]

    sample = read_csv(wp["sample"], "sample.csv (stage 00)")
    faces = read_csv(wp["faces_csv"], "faces.csv (stage 02)")
    masks = read_csv(cfg["paths"]["work_dir"] / "masks.csv", "masks.csv (stage 06)")
    feats = {m: E.load_feats(wp["features"] / f"{m}.npz") for m in (MD, MDM, BD)}
    df = sample.merge(faces, on="image_id", how="left").merge(masks[["image_id", "mask_frac", "mask_flag"]],
                                                              on="image_id", how="left")
    df["zoo"] = df["zoo"].fillna("unknown").astype(str)
    df["identity"] = df["identity"].astype(str)
    has = {m: df["image_id"].isin(feats[m].keys()) for m in feats}
    out = {"mask_stats": {"n": int(masks.shape[0]), "flagged": int(masks["mask_flag"].sum()),
                          "frac_median": float(masks["mask_frac"].median())}}

    # ------------------------------------------------ primary face-visible subset (same frames as stage 4)
    w = int(bcfg["primary_min_face_width_px"])
    mask = df["landmarks_ok"].fillna(False).astype(bool) & (df["face_w_px"] >= w) & has[BD] & has[MD]
    sub, _ = E.select_frames(df[mask], k, seed)
    assert has[MDM][df["image_id"].isin(sub["image_id"])].all(), "masked features missing for some subset frames"
    S = scores_for(sub, (MD, MDM, BD), feats)
    S["fused"] = (E.zscore_pairs(S[MD]) + E.zscore_pairs(S[BD])) / 2
    S["fused_masked"] = (E.zscore_pairs(S[MDM]) + E.zscore_pairs(S[BD])) / 2
    out["primary"] = evaluate(f"face_w>={w}px", sub, S, reps, seed)
    out["primary"]["flagged_masks"] = int(sub["mask_flag"].fillna(False).sum())

    # ------------------------------------------------ all frames, MegaDescriptor only (same frames as stage 4)
    sub_all, _ = E.select_frames(df[has[MD]], k, seed)
    S_all = scores_for(sub_all, (MD, MDM), feats)
    out["all_frames"] = evaluate("all_frames", sub_all, S_all, reps, seed, units=("sequence",))

    (res_dir / "masked.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    write_md(out, res_dir / "masked.md", reps)
    log.info("Done. See %s", res_dir / "masked.md")


def write_md(out, path, reps):
    ms = out["mask_stats"]
    L = ["# Background masking (stage 7)", "",
         f"SAM 2.1-B masks, prompted with the MegaDetector box; background set to mid-grey. "
         f"{ms['n']} crops, median mask = {ms['frac_median']:.0%} of the box, {ms['flagged']} flagged "
         "(mask under 5% or over 98% of the box). Designed after the stage-4 results; secondary.", ""]
    for key in ("primary", "all_frames"):
        r = out[key]
        L += [f"## {r['subset']} ({r['n_images']} images, {r['n_sequences']} sequences, {r['n_bears']} bears)", ""]
        models = list(next(iter(r["negatives"].values()))["auroc"])
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
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
