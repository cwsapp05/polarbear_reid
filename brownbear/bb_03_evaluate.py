"""Brown bear stage 3: MegaDescriptor vs BearID on the same head crops.

Pairs come from different encounters (bear-days) only. Three pair sets:
  all          every cross-encounter pair
  same_year    both photos from the same season (positives = same bear on another day)
  cross_year   photos from different seasons (positives = same bear after moult / weight change)
Each set restricts positives AND negatives, so its AUROC compares like with like.

Subsets:
  bearid_ok    frames where BearID found a face with eyes + nose (both models + fusion, paired)
  all_frames   every sampled frame (MegaDescriptor only: includes profiles BearID cannot use)

Paired cluster bootstrap over encounters and over bears. Fusion = equal-weight mean of z-scored
similarities (label-free), as in the PolarBearVidID run.

Output: results/summary.md, results/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics as M  # noqa: E402
from common import base_argparser, get_logger, load_config, read_csv  # noqa: E402

log = get_logger("bb.eval")
HERE = Path(__file__).resolve().parent
MD, BD, FU = "megadescriptor", "bearid", "fused"
NAMES = {MD: "MegaDescriptor", BD: "BearID", FU: "Fused"}
SIM = {MD: "cosine", BD: "neg_euclidean"}
DELTAS = [(BD, MD), (FU, MD), (FU, BD)]
PAIRSETS = ("all", "same_year", "cross_year")


def load_feats(path):
    z = np.load(path, allow_pickle=True)
    return dict(zip(z["image_ids"].tolist(), z["feats"]))


def zscore(S):
    v = S[np.triu_indices(len(S), k=1)]
    return (S - v.mean()) / v.std()


def pairs(df, which):
    enc, year, ident = (df[c].to_numpy() for c in ("encounter", "year", "identity"))
    i, j = np.triu_indices(len(df), k=1)
    keep = enc[i] != enc[j]
    if which == "same_year":
        keep &= year[i] == year[j]
    elif which == "cross_year":
        keep &= year[i] != year[j]
    i, j = i[keep].astype(np.int32), j[keep].astype(np.int32)
    return i, j, (ident[i] == ident[j]).astype(np.int8)


def ranking(S, df, cross_year=False):
    """Rank-1 and mAP; gallery = other encounters (cross_year: other seasons only)."""
    enc, year, ident = (df[c].to_numpy() for c in ("encounter", "year", "identity"))
    r1, aps = [], []
    for q in range(len(S)):
        g = (year != year[q]) if cross_year else (enc != enc[q])
        rel = ident[g] == ident[q]
        if not rel.any():
            continue
        r = rel[np.argsort(-S[q, g], kind="stable")]
        hits = np.flatnonzero(r)
        r1.append(r[0]); aps.append(np.mean((np.arange(len(hits)) + 1) / (hits + 1)))
    return {"rank1": float(np.mean(r1)), "mAP": float(np.mean(aps)), "n_queries": len(r1)}


def evaluate(name, df, S, reps, seed, units):
    out = {"subset": name, "n_images": len(df), "n_encounters": int(df["encounter"].nunique()),
           "n_bears": int(df["identity"].nunique()), "pairsets": {}}
    for ps in PAIRSETS:
        i, j, y = pairs(df, ps)
        sc = {m: s[i, j] for m, s in S.items()}
        r = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
             "auroc": {m: float(roc_auc_score(y, s)) for m, s in sc.items()}, "bootstrap": {}}
        for unit in units:
            codes, u = pd.factorize(df[unit])
            boot = M.paired_bootstrap(sc, y, codes[i], codes[j], len(u), reps, seed)
            b = {"ci95": {m: list(M.ci(v)) for m, v in boot.items()}, "deltas": {}}
            for a, c in DELTAS:
                if a in sc and c in sc:
                    d = boot[a] - boot[c]
                    b["deltas"][f"{a}|{c}"] = {"point": r["auroc"][a] - r["auroc"][c], "ci95": list(M.ci(d)),
                                               "p_bootstrap": M.bootstrap_p(d)}
            r["bootstrap"][unit] = b
        out["pairsets"][ps] = r
        log.info("[%s | %s] pos %d neg %d | %s", name, ps, r["n_pos"], r["n_neg"],
                 "  ".join(f"{NAMES[m]}: {v:.3f}" for m, v in r["auroc"].items()))
    out["ranking"] = {m: {"any_other_day": ranking(s, df), "other_season": ranking(s, df, True)} for m, s in S.items()}
    return out


def main() -> None:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work, res = cfg["paths"]["work_dir"], cfg["paths"]["results_dir"]
    res.mkdir(parents=True, exist_ok=True)
    reps, seed = int(cfg["evaluation"]["bootstrap_reps"]), int(cfg["evaluation"]["seed"])
    smp = read_csv(work / "sample.csv", "sample.csv")
    faces = read_csv(work / "faces.csv", "faces.csv (bb_01)")
    feats = {m: load_feats(work / "features" / f"{m}.npz") for m in (MD, BD)}
    df = smp.merge(faces, on="image_id", how="left")

    def sims(d, models):
        return {m: M.similarity(np.stack([feats[m][k] for k in d["image_id"]]), SIM[m]) for m in models}

    out = {"coverage": {"sampled": len(df), "bearid_embedded": int(df["image_id"].isin(feats[BD]).sum()),
                        "by_year": df.assign(ok=df["image_id"].isin(feats[BD])).groupby("year")["ok"].mean().round(3).to_dict()}}
    log.info("BearID coverage: %d / %d", out["coverage"]["bearid_embedded"], len(df))

    sub = df[df["image_id"].isin(feats[BD])].reset_index(drop=True)
    n_enc = sub.groupby("identity")["encounter"].nunique()
    sub = sub[sub["identity"].isin(n_enc[n_enc >= 2].index)].reset_index(drop=True)
    S = sims(sub, (MD, BD))
    S[FU] = (zscore(S[MD]) + zscore(S[BD])) / 2
    out["bearid_ok"] = evaluate("bearid_ok", sub, S, reps, seed, ("encounter", "identity"))

    out["all_frames"] = evaluate("all_frames", df, sims(df, (MD,)), reps, seed, ("encounter",))
    (res / "results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    write_md(out, res / "summary.md", reps)
    log.info("Done. See %s", res / "summary.md")


def fmt_p(p, reps):
    return f"<{1 / reps:.3g}" if p == 0 else f"{p:.3f}"


def write_md(out, path, reps):
    c = out["coverage"]
    L = ["# Brown bears (McNeil River heads): MegaDescriptor vs BearID", "",
         f"{c['sampled']} sampled head crops; BearID found a usable face (eyes + nose) in {c['bearid_embedded']} "
         f"({c['bearid_embedded'] / c['sampled']:.0%}). Pairs within one encounter (bear-day) are never scored.", ""]
    for key in ("bearid_ok", "all_frames"):
        r = out[key]
        models = list(r["pairsets"]["all"]["auroc"])
        unit0 = "encounter"
        L += [f"## {r['subset']} ({r['n_images']} images, {r['n_encounters']} encounters, {r['n_bears']} bears)", "",
              "| Pairs | Positives | Negatives | " + " | ".join(NAMES[m] + " AUROC [95% CI, encounter]" for m in models) + " |",
              "|---|---|---|" + "---|" * len(models)]
        for ps, n in r["pairsets"].items():
            ci = n["bootstrap"][unit0]["ci95"]
            L.append(f"| {ps.replace('_', '-')} | {n['n_pos']:,} | {n['n_neg']:,} | " + " | ".join(
                f"{n['auroc'][m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]" for m in models) + " |")
        if len(models) > 1:
            L += ["", "| Pairs | Comparison | Unit | Δ [95% CI] | p |", "|---|---|---|---|---|"]
            for ps, n in r["pairsets"].items():
                for unit, b in n["bootstrap"].items():
                    for k, d in b["deltas"].items():
                        a, cc = k.split("|")
                        L.append(f"| {ps.replace('_', '-')} | {NAMES[a]} − {NAMES[cc]} | {'bear' if unit == 'identity' else unit} | "
                                 f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {fmt_p(d['p_bootstrap'], reps)} |")
        L += ["", "| Model | Rank-1 (other days) | mAP (other days) | Rank-1 (other seasons) | mAP (other seasons) |",
              "|---|---|---|---|---|"]
        for m, v in r["ranking"].items():
            a, o = v["any_other_day"], v["other_season"]
            L.append(f"| {NAMES[m]} | {a['rank1']:.3f} | {a['mAP']:.3f} | {o['rank1']:.3f} | {o['mAP']:.3f} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
