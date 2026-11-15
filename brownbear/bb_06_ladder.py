"""Brown bear stage 6: the method ladder on the authors' leave-one-year-out splits.

For each held-out year Y (experiment test_on_Y):
  queries = sampled photos from year Y whose bear also appears in the training split
  gallery = sampled photos in train_iid of test_on_Y
Methods (all on identical queries and gallery):
  md            MegaDescriptor, zero-shot (cosine)
  bearid        BearID, zero-shot (negative Euclidean)
  md_head       small head trained on frozen MegaDescriptor features of the FULL train_iid split
                (CosFace loss; epoch picked on val_iid accuracy) -> "Swin trained on the population, no pose"
  md_head_faces / bearid_head   heads trained on the sampled training photos where BearID found a face
                (equal data for face vs whole-head features)
  poseswin      the authors' PoseSwin checkpoint for year Y (cosine)
Subsets:
  faces    queries and gallery restricted to photos where BearID found a face (every method can score)
  all      every sampled photo; a method that cannot embed a query (BearID: no face; PoseSwin-filtered:
           pose confidence < 0.8, as in the authors' pipeline) counts that query as a miss, and such
           photos are dropped from its gallery
Metrics: rank-1 and mAP per query, pooled over the six years; 95% CIs and paired deltas from a bootstrap
over bears (query identities). Also rank-1 by head angle (head_angle.csv). Secondary analysis.

Output: results/ladder.md, results/ladder.json, results/ladder_per_query.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import base_argparser, get_logger, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
log = get_logger("bb.ladder")
YEARS = range(2017, 2023)
NAMES = {"md": "MegaDescriptor (zero-shot)", "bearid": "BearID (zero-shot)",
         "md_head_faces": "MegaDescriptor + head (face subset)", "bearid_head": "BearID + head (face subset)",
         "md_head": "MegaDescriptor + head (full train split)", "poseswin": "PoseSwin",
         "poseswin_filtered": "PoseSwin, pose-filtered"}
DELTAS = [("bearid", "md"), ("md_head", "bearid"), ("poseswin", "md_head"), ("poseswin", "bearid"),
          ("bearid_head", "md_head_faces"), ("poseswin", "md")]
REPS, SEED = 1000, 0


def load(path):
    z = np.load(path, allow_pickle=True)
    d = dict(zip(z["image_ids"].tolist(), z["feats"]))
    return d, (dict(zip(z["image_ids"].tolist(), z["pose_conf"])) if "pose_conf" in z.files else None)


def ids_of(csv):
    return set(pd.read_csv(csv, usecols=["image"])["image"].str.replace(r"\.[^.]+$", "", regex=True))


# --------------------------------------------------------------------------- trained head
class Head(nn.Module):
    def __init__(self, d_in, n_cls, d=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.BatchNorm1d(d))
        self.W = nn.Parameter(torch.randn(n_cls, d) * 0.01)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)

    def logits(self, e, y=None, s=30.0, m=0.35):
        cos = e @ F.normalize(self.W, dim=1).T
        if y is not None:
            cos = cos - m * F.one_hot(y, cos.shape[1])
        return s * cos


def train_head(X, y, Xv=None, yv=None, epochs=40, seed=0):
    """CosFace head on frozen features; epoch chosen on validation accuracy (holdout 10% if none given)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    classes, y = np.unique(y, return_inverse=True)
    if Xv is None:
        perm = torch.randperm(len(X), generator=g).numpy()
        hv = perm[: len(X) // 10]
        mask = np.ones(len(X), bool); mask[hv] = False
        Xv, yv_idx, X, y = X[hv], y[hv], X[mask], y[mask]
    else:
        keep = np.isin(yv, classes)
        Xv, yv_idx = Xv[keep], np.searchsorted(classes, yv[keep])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, device=dev)
    Xvt = torch.tensor((Xv - mu) / sd, dtype=torch.float32, device=dev)
    yvt = torch.tensor(yv_idx, device=dev)
    h = Head(X.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(epochs):
        h.train()
        for b in torch.randperm(len(Xt), generator=g).split(256):
            b = b.to(dev)
            loss = F.cross_entropy(h.logits(h(Xt[b]), yt[b]), yt[b], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            acc = (h.logits(h(Xvt)).argmax(1) == yvt).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in h.state_dict().items()}
    h.load_state_dict(best_state)
    h.eval()

    def apply(Z):
        with torch.no_grad():
            return h(torch.tensor((Z - mu) / sd, dtype=torch.float32, device=dev)).cpu().numpy()
    return apply, best


# --------------------------------------------------------------------------- retrieval
def sim(Q, G, kind):
    if kind == "cosine":
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); G = G / np.linalg.norm(G, axis=1, keepdims=True)
        return Q @ G.T
    d2 = (Q * Q).sum(1)[:, None] + (G * G).sum(1)[None] - 2 * Q @ G.T
    return -np.sqrt(np.maximum(d2, 0))


def retrieve(S, q_id, g_id, q_date, g_date):
    """rank-1 and AP per query; gallery photos of the same bear on the same date are dropped."""
    r1, ap = np.zeros(len(S)), np.zeros(len(S))
    for i in range(len(S)):
        keep = ~((g_id == q_id[i]) & (g_date == q_date[i]))
        order = np.argsort(-S[i, keep], kind="stable")
        rel = (g_id[keep] == q_id[i])[order]
        if not rel.any():
            r1[i] = ap[i] = np.nan
            continue
        hits = np.flatnonzero(rel)
        r1[i], ap[i] = rel[0], np.mean((np.arange(len(hits)) + 1) / (hits + 1))
    return r1, ap


def score(method, Q, G, F_, kind, can_q, can_g):
    """Per-query rank-1 / AP; queries the method cannot embed count as misses (0)."""
    r1, ap = np.zeros(len(Q)), np.zeros(len(Q))
    qm, gm = Q["image_id"].map(can_q).to_numpy(bool), G["image_id"].map(can_g).to_numpy(bool)
    if qm.any() and gm.any():
        Qf = np.stack([F_[k] for k in Q.loc[qm, "image_id"]]); Gf = np.stack([F_[k] for k in G.loc[gm, "image_id"]])
        a, b = retrieve(sim(Qf, Gf, kind), Q.loc[qm, "identity"].to_numpy(), G.loc[gm, "identity"].to_numpy(),
                        Q.loc[qm, "date"].to_numpy(), G.loc[gm, "date"].to_numpy())
        r1[qm], ap[qm] = np.nan_to_num(a), np.nan_to_num(b)
    return r1, ap


# --------------------------------------------------------------------------- bootstrap over bears
def boot(per, methods, metric):
    rng = np.random.default_rng(SEED)
    bears = per["identity"].unique()
    grp = {b: per.index[per["identity"] == b].to_numpy() for b in bears}
    M = {m: per[f"{m}_{metric}"].to_numpy() for m in methods}
    out = {m: np.empty(REPS) for m in methods}
    for r in range(REPS):
        idx = np.concatenate([grp[b] for b in rng.choice(bears, len(bears))])
        for m in methods:
            out[m][r] = np.nanmean(M[m][idx])
    return out


def summarise(per, methods):
    res = {}
    for metric in ("r1", "ap"):
        b = boot(per, methods, metric)
        res[metric] = {"point": {m: float(np.nanmean(per[f"{m}_{metric}"])) for m in methods},
                       "ci95": {m: [float(np.quantile(b[m], .025)), float(np.quantile(b[m], .975))] for m in methods},
                       "deltas": {}}
        for a, c in DELTAS:
            if a in methods and c in methods:
                d = b[a] - b[c]
                res[metric]["deltas"][f"{a}|{c}"] = {
                    "point": res[metric]["point"][a] - res[metric]["point"][c],
                    "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                    "p": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return res


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--smoke", action="store_true", help="code test: sampled MD features stand in for the catalogue")
    args = ap.parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work, res = cfg["paths"]["work_dir"], cfg["paths"]["results_dir"]
    if args.smoke:
        global REPS
        REPS, res = 20, work / "smoke"
        res.mkdir(exist_ok=True)
    splits = cfg["paths"]["data_root"] / "data" / "reid_annotations"
    smp = pd.read_csv(work / "sample.csv")
    ang = pd.read_csv(work / "head_angle.csv")[["image_id", "head_angle"]]
    smp = smp.merge(ang, on="image_id", how="left")
    md, _ = load(work / "features" / "megadescriptor.npz")
    bd, _ = load(work / "features" / "bearid.npz")
    cat = md if args.smoke else load(work / "features" / "megadescriptor_catalogue.npz")[0]
    has_face = {k: k in bd for k in smp["image_id"]}
    always = {k: True for k in smp["image_id"]}

    rows, head_info = [], {}
    for Y in YEARS:
        tr, va = ids_of(splits / f"test_on_{Y}" / "train_iid.csv"), ids_of(splits / f"test_on_{Y}" / "val_iid.csv")
        G = smp[smp["image_id"].isin(tr)].reset_index(drop=True)
        Q = smp[(smp["year"] == Y) & smp["identity"].isin(G["identity"])].reset_index(drop=True)
        ps, pconf = load(work / "features" / f"poseswin_{Y}.npz")
        pose_ok = {k: pconf[k] >= 0.8 for k in smp["image_id"]}

        # step 3 heads
        trc = pd.read_csv(splits / f"test_on_{Y}" / "train_iid.csv", usecols=["image", "id"])
        trc["image_id"] = trc["image"].str.replace(r"\.[^.]+$", "", regex=True)
        vac = pd.read_csv(splits / f"test_on_{Y}" / "val_iid.csv", usecols=["image", "id"])
        vac["image_id"] = vac["image"].str.replace(r"\.[^.]+$", "", regex=True)
        trc, vac = trc[trc["image_id"].isin(cat)], vac[vac["image_id"].isin(cat)]
        f_md_head, acc = train_head(np.stack([cat[k] for k in trc["image_id"]]), trc["id"].to_numpy(),
                                    np.stack([cat[k] for k in vac["image_id"]]), vac["id"].to_numpy(), seed=Y)
        Gf = G[G["image_id"].map(has_face)]
        f_md_hf, acc2 = train_head(np.stack([md[k] for k in Gf["image_id"]]), Gf["identity"].to_numpy(), seed=Y)
        f_bd_h, acc3 = train_head(np.stack([bd[k] for k in Gf["image_id"]]), Gf["identity"].to_numpy(), seed=Y)
        head_info[Y] = {"md_head_val_acc": acc, "md_head_faces_holdout_acc": acc2, "bearid_head_holdout_acc": acc3,
                        "n_train_full": len(trc), "n_train_faces": len(Gf)}
        ids = smp["image_id"].tolist()
        feats = {"md": md, "bearid": bd, "poseswin": ps, "poseswin_filtered": ps,
                 "md_head": dict(zip(ids, f_md_head(np.stack([md[k] for k in ids])))),
                 "md_head_faces": dict(zip(ids, f_md_hf(np.stack([md[k] for k in ids])))),
                 "bearid_head": dict(zip(list(bd), f_bd_h(np.stack(list(bd.values())))))}
        kind = {m: ("euclid" if m == "bearid" else "cosine") for m in feats}
        can = {"md": always, "md_head": always, "poseswin": always, "poseswin_filtered": pose_ok,
               "bearid": has_face, "bearid_head": has_face, "md_head_faces": always}

        for subset in ("faces", "all"):
            if subset == "faces":
                q, g = Q[Q["image_id"].map(has_face)].reset_index(drop=True), G[G["image_id"].map(has_face)].reset_index(drop=True)
            else:
                q, g = Q, G
            per = q[["image_id", "identity", "head_angle"]].copy()
            per["year"], per["subset"] = Y, subset
            for m in feats:
                per[f"{m}_r1"], per[f"{m}_ap"] = score(m, q, g, feats[m], kind[m], can[m], can[m])
            rows.append(per)
        log.info("[%d] queries %d (faces %d), gallery %d | head val acc %.3f | r1 all: %s", Y, len(Q),
                 int(Q["image_id"].map(has_face).sum()), len(G), acc,
                 "  ".join(f"{m} {rows[-1][f'{m}_r1'].mean():.3f}" for m in feats))

    per = pd.concat(rows, ignore_index=True)
    per.to_csv(res / "ladder_per_query.csv", index=False)
    methods = list(NAMES)
    out = {"heads": head_info, "subsets": {}, "by_angle": {}, "per_year": {}}
    for subset in ("faces", "all"):
        p = per[per["subset"] == subset].reset_index(drop=True)
        out["subsets"][subset] = summarise(p, methods) | {"n_queries": len(p), "n_bears": int(p["identity"].nunique())}
        out["per_year"][subset] = {int(y): {m: float(d[f"{m}_r1"].mean()) for m in methods}
                                   for y, d in p.groupby("year")}
    pa = per[per["subset"] == "all"]
    for a, d in pa.groupby("head_angle"):
        d = d.reset_index(drop=True)
        out["by_angle"][a] = summarise(d, methods) | {"n_queries": len(d), "n_bears": int(d["identity"].nunique())}
    (res / "ladder.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    write_md(out, res / "ladder.md")
    log.info("Done. See %s", res / "ladder.md")


def write_md(out, path):
    L = ["# Method ladder on the leave-one-year-out splits (stage 6)", "",
         "Queries: sampled photos of the held-out year whose bear is in training; gallery: sampled train_iid photos. "
         "Pooled over 2017-2022. 95% CIs resample bears. Secondary analysis.", ""]
    for subset, r in out["subsets"].items():
        L += [f"## Subset: {subset} ({r['n_queries']} queries, {r['n_bears']} bears)", "",
              "| Method | Rank-1 [95% CI] | mAP [95% CI] |", "|---|---|---|"]
        for m in NAMES:
            a, b = r["r1"], r["ap"]
            L.append(f"| {NAMES[m]} | {a['point'][m]:.3f} [{a['ci95'][m][0]:.3f}, {a['ci95'][m][1]:.3f}] | "
                     f"{b['point'][m]:.3f} [{b['ci95'][m][0]:.3f}, {b['ci95'][m][1]:.3f}] |")
        L += ["", "| Comparison | Δ rank-1 [95% CI] | p | Δ mAP [95% CI] | p |", "|---|---|---|---|---|"]
        for k in r["r1"]["deltas"]:
            a, c = k.split("|")
            d1, d2 = r["r1"]["deltas"][k], r["ap"]["deltas"][k]
            L.append(f"| {NAMES[a]} − {NAMES[c]} | {d1['point']:+.3f} [{d1['ci95'][0]:+.3f}, {d1['ci95'][1]:+.3f}] | "
                     f"{d1['p']:.3f} | {d2['point']:+.3f} [{d2['ci95'][0]:+.3f}, {d2['ci95'][1]:+.3f}] | {d2['p']:.3f} |")
        L += ["", "Rank-1 by held-out year:", "", "| Year | " + " | ".join(NAMES[m] for m in NAMES) + " |",
              "|---|" + "---|" * len(NAMES)]
        for y, v in out["per_year"][subset].items():
            L.append(f"| {y} | " + " | ".join(f"{v[m]:.3f}" for m in NAMES) + " |")
        L.append("")
    L += ["## Rank-1 by head angle (subset: all; methods that cannot embed a photo score it as a miss)", "",
          "| Head angle | Queries | " + " | ".join(NAMES[m] for m in NAMES) + " |", "|---|---|" + "---|" * len(NAMES)]
    for a, r in out["by_angle"].items():
        L.append(f"| {a} | {r['n_queries']} | " + " | ".join(
            f"{r['r1']['point'][m]:.3f} [{r['r1']['ci95'][m][0]:.2f}, {r['r1']['ci95'][m][1]:.2f}]" for m in NAMES) + " |")
    L += ["", "Trained heads (validation accuracy used to pick the epoch):", ""]
    for y, h in out["heads"].items():
        L.append(f"- {y}: MD head val acc {h['md_head_val_acc']:.3f} on {h['n_train_full']} training photos; "
                 f"face-subset heads (holdout acc) MD {h['md_head_faces_holdout_acc']:.3f}, BearID {h['bearid_head_holdout_acc']:.3f} "
                 f"on {h['n_train_faces']} photos")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
