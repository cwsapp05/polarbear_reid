"""Pairwise verification AUROC with a paired cluster bootstrap, plus rank-1 / mAP."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- similarity
def similarity(feats: np.ndarray, kind: str) -> np.ndarray:
    f = feats.astype(np.float64)
    if kind == "cosine":
        f = f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-12)
        return f @ f.T
    if kind == "neg_euclidean":
        sq = (f * f).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * f @ f.T, 0)
        return -np.sqrt(d2)
    raise ValueError(f"unknown similarity {kind!r}")


# --------------------------------------------------------------------------- pairs
def make_pairs(identity, sequence, zoo, within_zoo: bool):
    """All image pairs i<j from different sequences. Returns (i, j, label).

    within_zoo=True keeps only negatives where both bears live in the same zoo, so a model can't
    score well just by recognising the enclosure, lighting or camera.
    """
    n = len(identity)
    i, j = np.triu_indices(n, k=1)
    keep = sequence[i] != sequence[j]
    if within_zoo:
        keep &= zoo[i] == zoo[j]
    i, j = i[keep].astype(np.int32), j[keep].astype(np.int32)
    return i, j, (identity[i] == identity[j]).astype(np.int8)


def roc(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------------- bootstrap
class WeightedAUC:
    """AUROC under arbitrary pair weights, with ties handled exactly (counted as 1/2).

    Scores are sorted once; each bootstrap replicate only changes the weights, so a replicate
    costs O(n_pairs) instead of a fresh sort.
    """

    def __init__(self, scores, labels, ui, uj):
        order = np.argsort(scores, kind="stable")
        s = scores[order]
        self.labels = labels[order].astype(bool)
        self.ui, self.uj = ui[order], uj[order]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])  # first index of each tie group

    def __call__(self, unit_weight: np.ndarray) -> float:
        w = unit_weight[self.ui] * unit_weight[self.uj]
        wp = np.where(self.labels, w, 0.0)
        wn = w - wp
        gp, gn = np.add.reduceat(wp, self.starts), np.add.reduceat(wn, self.starts)
        below = np.cumsum(gn) - gn
        tot_p, tot_n = gp.sum(), gn.sum()
        if tot_p == 0 or tot_n == 0:
            return np.nan
        return float((gp * (below + 0.5 * gn)).sum() / (tot_p * tot_n))


def paired_bootstrap(score_by_model: dict, labels, unit_i, unit_j, n_units, reps, seed):
    """Cluster bootstrap: resample units (sequences or bears) with replacement; a pair's weight is
    the product of its two units' multiplicities. Same draws for every model -> paired deltas."""
    rng = np.random.default_rng(seed)
    aucs = {m: WeightedAUC(s, labels, unit_i, unit_j) for m, s in score_by_model.items()}
    out = {m: np.empty(reps) for m in aucs}
    for r in range(reps):
        mult = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units)).astype(np.float64)
        for m, f in aucs.items():
            out[m][r] = f(mult)
    return out


def ci(x, level=0.95):
    x = x[np.isfinite(x)]
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def bootstrap_p(delta):
    """Two-sided bootstrap p-value for H0: delta = 0 (percentile method)."""
    d = delta[np.isfinite(delta)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


# --------------------------------------------------------------------------- ranking
def rank_metrics(S: np.ndarray, identity, sequence):
    """Leave-one-sequence-out retrieval: each image queries every image from OTHER sequences."""
    rank1, aps = [], []
    for q in range(len(S)):
        g = sequence != sequence[q]
        rel = identity[g] == identity[q]
        if not rel.any():
            continue
        order = np.argsort(-S[q, g], kind="stable")
        r = rel[order]
        rank1.append(r[0])
        hits = np.flatnonzero(r)
        aps.append(np.mean((np.arange(len(hits)) + 1) / (hits + 1)))
    return {"rank1": float(np.mean(rank1)), "mAP": float(np.mean(aps)), "n_queries": len(aps)}
