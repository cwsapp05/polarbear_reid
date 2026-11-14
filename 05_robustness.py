"""Stage 5: robustness checks on the stage-4 comparison (primary face-visible subset).

Runs on exactly the frames stage 4 used for the headline subset (same selection, same seed):

  A. Composition   - how much each bear contributes to the subset and to the positive pairs.
  B. Bootstrap unit - sequence-level vs individual-level (bear) cluster bootstrap, side by side.
  C. Bear balance  - AUROC with every bear (positives) and every bear pair (negatives) weighted
                     equally, plus leave-one-bear-out deltas, so a few camera-facing bears
                     can't drive the result.
  D. Tuned fusion  - fused = w * z(MD) + (1 - w) * z(BearID). w is tuned leave-one-zoo-out: for
                     each zoo, w maximises AUROC on pairs that don't touch that zoo, then scores
                     that zoo's pairs. Out-of-fold scores are pooled and compared with each model
                     and with the untuned equal-weight fusion on the same pairs.

Output: results/robustness.md, results/robustness.json
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

log = get_logger("robustness")
_spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).with_name("04_evaluate.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
MODELS, FUSED = E.MODELS, E.FUSED
W_GRID = np.round(np.linspace(0, 1, 21), 2)  # weight on MegaDescriptor


def load_subset(cfg):
    wp, ecfg, bcfg = work_paths(cfg), cfg["evaluation"], cfg["bearid"]
    sample = read_csv(wp["sample"], "sample.csv (stage 00)")
    faces = read_csv(wp["faces_csv"], "faces.csv (stage 02)")
    feats = {m: E.load_feats(wp["features"] / f"{m}.npz") for m in MODELS}
    df = sample.merge(faces, on="image_id", how="left")
    df["zoo"] = df["zoo"].fillna("unknown").astype(str)
    df["identity"] = df["identity"].astype(str)
    w = int(bcfg["primary_min_face_width_px"])
    mask = (df["landmarks_ok"].fillna(False).astype(bool) & (df["face_w_px"] >= w)
            & df["image_id"].isin(feats["bearid"].keys()) & df["image_id"].isin(feats["megadescriptor"].keys()))
    sub, dropped = E.select_frames(df[mask], int(ecfg["max_frames_per_sequence"]), int(ecfg["seed"]))
    Z = {m: E.zscore_pairs(M.similarity(np.stack([feats[m][i] for i in sub["image_id"]]), ecfg["similarity"][m]))
         for m in MODELS}
    return f"face_w>={w}px", sub, dropped, Z


def boot_summary(scores, y, codes_i, codes_j, n_units, reps, seed):
    boot = M.paired_bootstrap(scores, y, codes_i, codes_j, n_units, reps, seed)
    out = {m: {"auroc": float(roc_auc_score(y, s)), "ci95": list(M.ci(boot[m]))} for m, s in scores.items()}
    for a, b in E.DELTAS:
        if a in scores and b in scores:
            d = boot[a] - boot[b]
            out[f"{a}-{b}"] = {"point": out[a]["auroc"] - out[b]["auroc"], "ci95": list(M.ci(d)),
                               "p_bootstrap": M.bootstrap_p(d)}
    return out


def balanced_weights(ident, i, j, y):
    """Each bear's positive pairs sum to 1; each (unordered) bear pair's negatives sum to 1."""
    a, b = ident[i], ident[j]
    codes, _ = pd.factorize(ident)
    ca, cb = codes[i], codes[j]
    n = codes.max() + 1
    key = np.where(y == 1, ca, n + np.minimum(ca, cb) * n + np.maximum(ca, cb))
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def tuned_fusion(Z, ident, seq, zoo, mode):
    """Leave-one-zoo-out weight tuning. Returns out-of-fold scores on test pairs + chosen weights."""
    i, j, y = M.make_pairs(ident, seq, zoo, within_zoo=(mode == "within_zoo"))
    zi, zj = zoo[i], zoo[j]
    sm, sb = Z["megadescriptor"][i, j], Z["bearid"][i, j]
    oof = np.full(len(y), np.nan)
    folds = {}
    for z in np.unique(zoo):
        test = (zi == z) | (zj == z)
        train = ~test
        aucs = [roc_auc_score(y[train], w * sm[train] + (1 - w) * sb[train]) for w in W_GRID]
        w_best = float(W_GRID[int(np.argmax(aucs))])
        oof[test] = w_best * sm[test] + (1 - w_best) * sb[test]
        folds[z] = {"w_megadescriptor": w_best, "train_auroc": float(max(aucs)),
                    "n_test_pairs": int(test.sum())}
    # Cross-zoo pairs touch two folds, so oof holds whichever fold ran last; the caller rescores them.
    return i, j, y, oof, folds


def main() -> None:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ecfg = cfg["evaluation"]
    reps, seed = int(ecfg["bootstrap_reps"]), int(ecfg["seed"])
    res_dir = cfg["paths"]["results_dir"]
    name, sub, dropped, Z = load_subset(cfg)
    ident, seq, zoo = (sub[c].to_numpy().astype(str) for c in ("identity", "sequence", "zoo"))
    S = dict(Z)
    S[FUSED] = (Z["megadescriptor"] + Z["bearid"]) / 2  # same ranking as stage 4's fused score
    log.info("%s: %d images, %d sequences, %d bears (dropped: %s)", name, len(sub), sub["sequence"].nunique(),
             sub["identity"].nunique(), dropped)
    out = {"subset": name, "n_images": len(sub), "dropped_bears": dropped}

    # ---------------------------------------------------------------- A. composition
    i_all, j_all, y_all = M.make_pairs(ident, seq, zoo, within_zoo=False)
    pos_by_bear = pd.Series(ident[i_all[y_all == 1]]).value_counts()
    comp = (sub.groupby(["zoo", "identity"]).agg(images=("image_id", "size"), sequences=("sequence", "nunique"))
            .reset_index())
    comp["image_share"] = comp["images"] / comp["images"].sum()
    comp["pos_pair_share"] = comp["identity"].map(pos_by_bear).fillna(0) / pos_by_bear.sum()
    comp = comp.sort_values("pos_pair_share", ascending=False)
    out["composition"] = comp.to_dict(orient="records")

    # ---------------------------------------------------------------- B/C. per negative mode
    out["modes"] = {}
    for mode in ("all", "within_zoo"):
        i, j, y = M.make_pairs(ident, seq, zoo, within_zoo=(mode == "within_zoo"))
        scores = {m: S[m][i, j] for m in S}
        r = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "bootstrap": {}}
        for unit in ("sequence", "identity"):
            codes, units = pd.factorize(sub[unit])
            r["bootstrap"][unit] = boot_summary(scores, y, codes[i], codes[j], len(units), reps, seed)
            log.info("[%s | %s bootstrap] Δ BearID−MD %+.3f [%+.3f, %+.3f]  Δ fused−MD %+.3f [%+.3f, %+.3f]",
                     mode, unit, *[v for k in ("bearid-megadescriptor", "fused-megadescriptor")
                                   for v in (r["bootstrap"][unit][k]["point"], *r["bootstrap"][unit][k]["ci95"])])

        w = balanced_weights(ident, i, j, y)
        r["balanced"] = {m: float(roc_auc_score(y, s, sample_weight=w)) for m, s in scores.items()}

        lobo = []
        for bear in np.unique(ident):
            keep = (ident[i] != bear) & (ident[j] != bear)
            yk = y[keep]
            if yk.sum() == 0 or (1 - yk).sum() == 0:
                continue
            a = {m: float(roc_auc_score(yk, s[keep])) for m, s in scores.items()}
            lobo.append({"dropped": bear, **a, "bearid-megadescriptor": a["bearid"] - a["megadescriptor"],
                         "fused-megadescriptor": a[FUSED] - a["megadescriptor"]})
        r["leave_one_bear_out"] = lobo

        # ------------------------------------------------------------ D. tuned fusion
        ti, tj, ty, oof, folds = tuned_fusion(Z, ident, seq, zoo, mode)
        assert (ti == i).all() and (tj == j).all()
        same_zoo = zoo[i] == zoo[j]
        tf = {"folds": folds, "w_grid": W_GRID.tolist()}
        # Within-zoo pairs belong to exactly one fold, so their OOF score is clean. Cross-zoo pairs
        # touch two held-out zoos; score them by averaging the two folds' weights (still held out).
        if mode == "all":
            wz = {z: f["w_megadescriptor"] for z, f in folds.items()}
            wpair = (np.vectorize(wz.get)(zoo[i]) + np.vectorize(wz.get)(zoo[j])) / 2
            oof = np.where(same_zoo, oof, wpair * scores["megadescriptor"] + (1 - wpair) * scores["bearid"])
        sc = {"megadescriptor": scores["megadescriptor"], "bearid": scores["bearid"], FUSED: scores[FUSED],
              "tuned": oof}
        codes, units = pd.factorize(sub["sequence"])
        boot = M.paired_bootstrap(sc, y, codes[i], codes[j], len(units), reps, seed)
        tf["auroc"] = {m: {"auroc": float(roc_auc_score(y, s)), "ci95": list(M.ci(boot[m]))} for m, s in sc.items()}
        for b in ("megadescriptor", "bearid", FUSED):
            d = boot["tuned"] - boot[b]
            tf[f"tuned-{b}"] = {"point": tf["auroc"]["tuned"]["auroc"] - tf["auroc"][b]["auroc"],
                                "ci95": list(M.ci(d)), "p_bootstrap": M.bootstrap_p(d)}
        r["tuned_fusion"] = tf
        log.info("[%s] tuned fusion OOF %.3f (weights on MD by held-out zoo: %s)", mode,
                 tf["auroc"]["tuned"]["auroc"], {z: f["w_megadescriptor"] for z, f in folds.items()})
        out["modes"][mode] = r

    (res_dir / "robustness.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    write_md(out, res_dir / "robustness.md", reps)
    log.info("Done. See %s", res_dir / "robustness.md")


def _ci(d):
    return f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def _delta(d, reps):
    p = E.fmt_p(d["p_bootstrap"], reps)
    return f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", p


def write_md(out, path, reps):
    L = [f"# Robustness checks ({out['subset']})", "",
         f"Same {out['n_images']} frames as the stage-4 headline subset. "
         "All of this was planned after the stage-4 results were seen, so treat it as secondary.", ""]

    L += ["## A. Who is in the face-visible subset", "",
          "| Zoo | Bear | Images | Seq. | Image share | Positive-pair share |", "|---|---|---|---|---|---|"]
    for c in out["composition"]:
        L.append(f"| {c['zoo']} | {c['identity']} | {c['images']} | {c['sequences']} | "
                 f"{c['image_share']:.1%} | {c['pos_pair_share']:.1%} |")
    top3 = sum(c["pos_pair_share"] for c in out["composition"][:3])
    L += ["", f"The top three bears account for {top3:.0%} of positive pairs.", ""]

    L += ["## B. Sequence-level vs bear-level bootstrap", "",
          "| Negatives | Bootstrap unit | MegaDescriptor | BearID | Fused | Δ BearID − MD | p | Δ Fused − MD | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for mode, r in out["modes"].items():
        for unit, b in r["bootstrap"].items():
            d1, p1 = _delta(b["bearid-megadescriptor"], reps)
            d2, p2 = _delta(b["fused-megadescriptor"], reps)
            L.append(f"| {mode.replace('_', '-')} | {unit if unit == 'sequence' else 'bear'} | "
                     f"{_ci(b['megadescriptor'])} | {_ci(b['bearid'])} | {_ci(b[FUSED])} | {d1} | {p1} | {d2} | {p2} |")
    L += ["", "With only 13 bears, the bear-level intervals are the honest ones for generalising to new bears; "
          "the sequence-level intervals only cover new footage of these 13 bears.", ""]

    L += ["## C. Bear-balanced AUROC and leave-one-bear-out", "",
          "Balanced = every bear's positive pairs and every bear pair's negatives carry equal total weight.", "",
          "| Negatives | MegaDescriptor | BearID | Fused | Δ BearID − MD | Δ Fused − MD |", "|---|---|---|---|---|---|"]
    for mode, r in out["modes"].items():
        b = r["balanced"]
        L.append(f"| {mode.replace('_', '-')} | {b['megadescriptor']:.3f} | {b['bearid']:.3f} | {b[FUSED]:.3f} | "
                 f"{b['bearid'] - b['megadescriptor']:+.3f} | {b[FUSED] - b['megadescriptor']:+.3f} |")
    L += ["", "Leave-one-bear-out (point estimates; range over the 13 drops):", ""]
    for mode, r in out["modes"].items():
        lo = pd.DataFrame(r["leave_one_bear_out"])
        for k in ("bearid-megadescriptor", "fused-megadescriptor"):
            a, b = lo[k].idxmin(), lo[k].idxmax()
            L.append(f"- {mode.replace('_', '-')}, Δ {k.replace('-', ' − ').replace('megadescriptor', 'MD')}: "
                     f"{lo[k].min():+.3f} (without {lo.loc[a, 'dropped']}) to {lo[k].max():+.3f} "
                     f"(without {lo.loc[b, 'dropped']})")
    L.append("")

    L += ["## D. Fusion weight tuned leave-one-zoo-out", "",
          "Weight on MegaDescriptor chosen from {0, 0.05, …, 1} on the other five zoos, applied to the held-out zoo. "
          "Bootstrap is over sequences and conditional on the chosen weights.", ""]
    for mode, r in out["modes"].items():
        tf = r["tuned_fusion"]
        ws = ", ".join(f"{z}: {f['w_megadescriptor']:.2f}" for z, f in tf["folds"].items())
        L += [f"**{mode.replace('_', '-')} negatives.** Weights on MD per held-out zoo: {ws}.", "",
              "| Score | AUROC [95% CI] | Δ tuned − this [95% CI] | p |", "|---|---|---|---|"]
        for m in ("megadescriptor", "bearid", FUSED, "tuned"):
            if m == "tuned":
                L.append(f"| tuned (out-of-fold) | {_ci(tf['auroc'][m])} | – | – |")
            else:
                d, p = _delta(tf[f"tuned-{m}"], reps)
                L.append(f"| {m if m != FUSED else 'fused (equal weight)'} | {_ci(tf['auroc'][m])} | {d} | {p} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
