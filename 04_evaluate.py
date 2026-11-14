"""Stage 4: compare MegaDescriptor and BearID by pairwise verification AUROC.

For each face-visible subset (one per minimum face width) both models are scored on the SAME
frames and the SAME pairs:
  * pairs = every two frames from different sequences (same-sequence pairs are near-duplicates)
  * positive = same bear, negative = different bears
  * score = each model's native similarity
Two negative sets are reported: all negatives, and within-zoo negatives only (harder; removes
enclosure/camera cues). Uncertainty comes from a paired cluster bootstrap over sequences.
MegaDescriptor is also scored on all frames as a secondary result. When evaluation.fusion is
enabled, a fused score (equal-weight mean of the two z-scored similarities) is scored on the
face-visible subsets as a third, secondary model.

Output: results/results.json, results/summary.md, results/results_table.csv, results/*.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import metrics as M
import plotting as P
from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("evaluate")
MODELS = ("megadescriptor", "bearid")
FUSED = "fused"
FUSED_METRIC = "mean z-score"
# (a, b) -> key "delta_<a>_minus_<b>" in results.json
DELTAS = (("bearid", "megadescriptor"), (FUSED, "megadescriptor"), (FUSED, "bearid"))


def zscore_pairs(S):
    """Standardise a similarity matrix by the mean/std of its off-diagonal pairs (label-free)."""
    v = S[np.triu_indices(len(S), k=1)]
    return (S - v.mean()) / v.std()


def metric_name(m, ecfg):
    return FUSED_METRIC if m == FUSED else ecfg["similarity"][m]


def load_feats(path):
    z = np.load(path, allow_pickle=True)
    return dict(zip(z["image_ids"].tolist(), z["feats"]))


def select_frames(df, k, seed):
    """Up to k random frames per sequence; drop bears seen in only one sequence (no positives)."""
    df = df.sample(frac=1.0, random_state=seed).groupby("sequence").head(k)
    n_seq = df.groupby("identity")["sequence"].nunique()
    dropped = n_seq[n_seq < 2].index.tolist()
    return df[~df["identity"].isin(dropped)].reset_index(drop=True), dropped


def evaluate_subset(name, df, models, feats, ecfg, res_dir, make_figs):
    ident, seq, zoo = (df[c].to_numpy() for c in ("identity", "sequence", "zoo"))
    S = {m: M.similarity(np.stack([feats[m][i] for i in df["image_id"]]), ecfg["similarity"][m]) for m in models}
    if ecfg.get("fusion", {}).get("enabled") and set(MODELS) <= set(models):
        S[FUSED] = sum(zscore_pairs(S[m]) for m in MODELS) / len(MODELS)
        models = tuple(models) + (FUSED,)

    unit_col = "sequence" if ecfg["bootstrap_unit"] == "sequence" else "identity"
    unit_codes, units = pd.factorize(df[unit_col])
    out = {"subset": name, "n_images": len(df), "n_sequences": int(df["sequence"].nunique()),
           "n_bears": int(df["identity"].nunique()), "n_zoos": int(df["zoo"].nunique()), "negatives": {}}

    neg_modes = ["all"] + (["within_zoo"] if df["zoo"].nunique() > 1 and (df["zoo"] != "unknown").all() else [])
    for mode in neg_modes:
        i, j, y = M.make_pairs(ident, seq, zoo, within_zoo=(mode == "within_zoo"))
        if y.sum() == 0 or (1 - y).sum() == 0:
            log.warning("[%s/%s] no positive or no negative pairs - skipped", name, mode)
            continue
        scores = {m: S[m][i, j] for m in models}
        boot = M.paired_bootstrap(scores, y, unit_codes[i], unit_codes[j], len(units),
                                  int(ecfg["bootstrap_reps"]), int(ecfg["seed"]))
        r = {"n_pos_pairs": int(y.sum()), "n_neg_pairs": int((1 - y).sum()), "models": {}}
        curves = {}
        for m in models:
            fpr, tpr, auc = M.roc(scores[m], y)
            lo, hi = M.ci(boot[m])
            r["models"][m] = {"auroc": auc, "ci95": [lo, hi]}
            curves[m] = (fpr, tpr, auc, (lo, hi))
        for a, b in DELTAS:
            if a in models and b in models:
                delta = boot[a] - boot[b]
                point = r["models"][a]["auroc"] - r["models"][b]["auroc"]
                r[f"delta_{a}_minus_{b}"] = {"point": point, "ci95": list(M.ci(delta)),
                                             "p_bootstrap": M.bootstrap_p(delta)}
        out["negatives"][mode] = r
        if make_figs:
            tag = f"{name}_{mode}".replace(">=", "ge")  # '>' is not allowed in Windows file names
            P.roc_curves(curves, f"ROC: {name}, {mode.replace('_', '-')} negatives", res_dir / f"roc_{tag}.png")
            P.similarity_hists({m: (scores[m][y == 1], scores[m][y == 0], metric_name(m, ecfg)) for m in models},
                               f"Pair similarity: {name}, {mode.replace('_', '-')} negatives",
                               res_dir / f"similarity_{tag}.png")
        log.info("[%s | %s negatives] %s", name, mode, "  ".join(
            f"{m}: {r['models'][m]['auroc']:.3f} [{r['models'][m]['ci95'][0]:.3f}, {r['models'][m]['ci95'][1]:.3f}]"
            for m in models))

    if ecfg.get("rank_metrics", True):
        out["ranking"] = {m: M.rank_metrics(S[m], ident, seq) for m in models}
    return out


def fmt_ci(d):
    return f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def fmt_p(p, reps):
    return f"<{1 / reps:.3g}" if p == 0 else f"{p:.3f}"


def fmt_delta(n, key, reps):
    if key not in n:
        return "–", "–"
    d = n[key]
    return f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", fmt_p(d["p_bootstrap"], reps)


def write_summary(results, primary, path, reps):
    fused = any(FUSED in n["models"] for r in results for n in r["negatives"].values())
    lines = ["# PolarBearVidID: MegaDescriptor (body) vs BearID (face)", "",
             "Verification AUROC, pairs from different sequences only, 95% paired cluster-bootstrap CIs.", ""]
    if fused:
        lines += ["*Fused* = equal-weight mean of the two models' z-scored similarities (untuned, label-free "
                  "normalisation). It was added after the first results showed the models' errors overlap "
                  "little, so treat it as a secondary analysis.", ""]
    for mode in ("all", "within_zoo"):
        rows = [r for r in results if mode in r["negatives"]]
        if not rows:
            continue
        head = "| Subset | Images | Seq. | Bears | MegaDescriptor | BearID | Δ (BearID − MD) [95% CI] | p |"
        if fused:
            head += " Fused | Δ (Fused − MD) [95% CI] | p |"
        lines += [f"## Negatives: {mode.replace('_', '-')}", "", head, "|" + "---|" * (head.count("|") - 1)]
        for r in rows:
            n = r["negatives"][mode]
            md = fmt_ci(n["models"]["megadescriptor"])
            bd = fmt_ci(n["models"]["bearid"]) if "bearid" in n["models"] else "–"
            ds, p = fmt_delta(n, "delta_bearid_minus_megadescriptor", reps)
            row = f"| {r['subset']} | {r['n_images']} | {r['n_sequences']} | {r['n_bears']} | {md} | {bd} | {ds} | {p} |"
            if fused:
                fu = fmt_ci(n["models"][FUSED]) if FUSED in n["models"] else "–"
                fds, fp = fmt_delta(n, f"delta_{FUSED}_minus_megadescriptor", reps)
                row += f" {fu} | {fds} | {fp} |"
            lines.append(row)
        lines.append("")

    prim = next((r for r in results if r["subset"] == primary), None)
    if prim and "all" in prim["negatives"] and "delta_bearid_minus_megadescriptor" in prim["negatives"]["all"]:
        lines += [f"## Headline ({primary})", ""]
        for mode in ("all", "within_zoo"):
            if mode not in prim["negatives"]:
                continue
            n = prim["negatives"][mode]
            d = n["delta_bearid_minus_megadescriptor"]
            lo, hi = d["ci95"]
            if lo > 0:
                verdict = "BearID scores higher, and the interval excludes zero"
            elif hi < 0:
                verdict = "MegaDescriptor scores higher, and the interval excludes zero"
            else:
                verdict = "the difference is not distinguishable from zero at this sample size"
            lines.append(f"- {mode.replace('_', '-')} negatives: MegaDescriptor {fmt_ci(n['models']['megadescriptor'])}, "
                         f"BearID {fmt_ci(n['models']['bearid'])}; Δ = {d['point']:+.3f} [{lo:+.3f}, {hi:+.3f}], "
                         f"p {fmt_p(d['p_bootstrap'], reps).replace('<', '< ') if d['p_bootstrap'] == 0 else '= ' + fmt_p(d['p_bootstrap'], reps)} - {verdict}.")
        if fused:
            lines += ["", "Secondary (fused score):"]
            for mode in ("all", "within_zoo"):
                n = prim["negatives"].get(mode, {})
                if FUSED not in n.get("models", {}):
                    continue
                parts = []
                for b in MODELS:
                    ds, p = fmt_delta(n, f"delta_{FUSED}_minus_{b}", reps)
                    parts.append(f"Δ vs {'MegaDescriptor' if b == 'megadescriptor' else 'BearID'} = {ds}, p {p}")
                lines.append(f"- {mode.replace('_', '-')} negatives: fused {fmt_ci(n['models'][FUSED])}; " + "; ".join(parts) + ".")
        lines.append("")

    if any("ranking" in r for r in results):
        lines += ["## Retrieval (leave-one-sequence-out)", "", "| Subset | Model | Rank-1 | mAP |", "|---|---|---|---|"]
        for r in results:
            for m, v in r.get("ranking", {}).items():
                lines.append(f"| {r['subset']} | {m} | {v['rank1']:.3f} | {v['mAP']:.3f} |")
        lines.append("")
    lines += [("The fused score only exists on face-visible frames (see the stage 2 log for how many); "
               "elsewhere a hybrid falls back to MegaDescriptor alone. " if fused else "") +
              "Caveats: 13 zoo bears from 6 institutions; both models are used zero-shot. BearID's networks "
              "were trained on brown bears, so this measures transfer to polar bears, not BearID's ceiling.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp = work_paths(cfg)
    ecfg, bcfg = cfg["evaluation"], cfg["bearid"]
    res_dir = cfg["paths"]["results_dir"]

    sample = read_csv(wp["sample"], "sample.csv (stage 00)")
    faces = read_csv(wp["faces_csv"], "faces.csv (stage 02)")
    feats = {m: load_feats(wp["features"] / f"{m}.npz") for m in MODELS}
    df = sample.merge(faces, on="image_id", how="left")
    df["zoo"] = df["zoo"].fillna("unknown").astype(str)
    df["identity"] = df["identity"].astype(str)
    has = {m: df["image_id"].isin(feats[m].keys()) for m in MODELS}
    k, seed = int(ecfg["max_frames_per_sequence"]), int(ecfg["seed"])
    primary = f"face_w>={int(bcfg['primary_min_face_width_px'])}px"

    results, by_w = [], []
    for w in [int(x) for x in bcfg["min_face_widths_px"]]:
        name = f"face_w>={w}px"
        mask = df["landmarks_ok"].fillna(False).astype(bool) & (df["face_w_px"] >= w) & has["bearid"] & has["megadescriptor"]
        sub, dropped = select_frames(df[mask], k, seed)
        if dropped:
            log.warning("[%s] bears with <2 face-visible sequences dropped: %s", name, dropped)
        if sub["identity"].nunique() < 2:
            log.warning("[%s] fewer than 2 bears left - skipped", name)
            continue
        r = evaluate_subset(name, sub, MODELS, feats, ecfg, res_dir, make_figs=(name == primary))
        r["dropped_bears"] = dropped
        results.append(r)
        for m, d in r["negatives"].get("all", {}).get("models", {}).items():
            by_w.append({"subset_w": w, "model": m, "auc": d["auroc"], "lo": d["ci95"][0], "hi": d["ci95"][1]})

    # secondary: MegaDescriptor on every frame (no face requirement)
    sub, _ = select_frames(df[has["megadescriptor"]], k, seed)
    results.append(evaluate_subset("all_frames", sub, ("megadescriptor",), feats, ecfg, res_dir, make_figs=True))

    if by_w:
        P.auc_by_threshold(by_w, res_dir / "auroc_by_face_width.png")
    (res_dir / "results.json").write_text(json.dumps({"config": {k_: v for k_, v in cfg.items() if k_ != "paths"},
                                                      "results": results}, indent=2, default=str))
    flat = []
    for r in results:
        for mode, n in r["negatives"].items():
            for m, v in n["models"].items():
                flat.append({"subset": r["subset"], "negatives": mode, "model": m, "auroc": v["auroc"],
                             "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1], "n_images": r["n_images"],
                             "n_sequences": r["n_sequences"], "n_bears": r["n_bears"],
                             "n_pos_pairs": n["n_pos_pairs"], "n_neg_pairs": n["n_neg_pairs"],
                             **{k_: r.get("ranking", {}).get(m, {}).get(k_) for k_ in ("rank1", "mAP")}})
    pd.DataFrame(flat).to_csv(res_dir / "results_table.csv", index=False)
    write_summary(results, primary, res_dir / "summary.md", int(ecfg["bootstrap_reps"]))
    log.info("Done. See %s", res_dir / "summary.md")


if __name__ == "__main__":
    main()
