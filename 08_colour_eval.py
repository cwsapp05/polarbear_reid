"""Stage 8: is MegaDescriptor's cross-zoo lead carried by colour, brightness or contrast?

Stage 7 showed the lead over BearID survives background masking, so whatever separates zoos sits on
the bear. This stage re-embeds crops with MegaDescriptor after removing colour information:

  grey          luminance only (ITU-R 601), copied to all three channels
  cnorm         grey-world colour normalisation: each channel shifted to mean 128 (removes colour
                cast and brightness), then all channels scaled by one shared factor to std 50
                (removes contrast differences, keeps relative hue)
  masked_grey   the same on the SAM-masked crops (stage 6); statistics use bear pixels only and
  masked_cnorm  the grey background stays at 128

If colour cast / exposure drove the lead, these variants should lose more AUROC with ALL negatives
than with WITHIN-ZOO negatives, i.e. the gap AUROC(all) - AUROC(within-zoo) should shrink.
Bootstraps use the same seed and units in both negative sets, so the gap change gets a paired CI.

Only frames that enter the evaluation are embedded (same selection and seed as stages 4 and 7).
Designed after the stage-7 results; secondary analysis.

Output: work/features/megadescriptor_{variant}.npz, results/colour.md, results/colour.json,
        results/colour_examples.png
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import metrics as M
from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("colour")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("evaluate", "04_evaluate.py")

GREY, TARGET_STD = 128.0, 50.0
MD, MDM, BD = "megadescriptor", "megadescriptor_masked", "bearid"
VARIANTS = {"grey": (False, "grey"), "cnorm": (False, "cnorm"),
            "masked_grey": (True, "grey"), "masked_cnorm": (True, "cnorm")}   # name -> (masked input, op)
FEAT = {v: f"megadescriptor_{v}" for v in VARIANTS}
NAMES = {MD: "MD", FEAT["grey"]: "MD grey", FEAT["cnorm"]: "MD colour-norm", MDM: "MD masked",
         FEAT["masked_grey"]: "MD masked+grey", FEAT["masked_cnorm"]: "MD masked+colour-norm", BD: "BearID"}
SIM = {m: ("neg_euclidean" if m == BD else "cosine") for m in NAMES}
DELTAS = [(FEAT["grey"], MD), (FEAT["cnorm"], MD), (FEAT["masked_grey"], MDM), (FEAT["masked_cnorm"], MDM),
          (FEAT["masked_cnorm"], MD), (BD, MD), (BD, FEAT["cnorm"]), (BD, FEAT["masked_cnorm"])]


def transform_array(img: np.ndarray, masked: bool, op: str) -> np.ndarray:
    """img: HxWx3 uint8. For masked crops the background is exactly (128,128,128) and stays so."""
    x = img.astype(np.float32)
    fg = ~np.all(img == GREY, axis=-1) if masked else np.ones(img.shape[:2], bool)
    if fg.sum() < 16:          # empty mask: nothing to normalise
        return img
    if op == "grey":
        y = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = np.repeat(y[..., None], 3, axis=-1)
    elif op == "cnorm":
        px = x[fg]
        mu = px.mean(0)
        sd = (px - mu).std()
        x = (x - mu) * (TARGET_STD / max(sd, 1e-3)) + GREY
    else:
        raise ValueError(op)
    x[~fg] = GREY
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


class Variant:
    """Map-style dataset; module-level so Windows DataLoader workers can pickle it."""

    def __init__(self, paths, masked, op, transform):
        self.paths, self.masked, self.op, self.transform = paths, masked, op, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.paths[i]).convert("RGB"))
        return self.transform(Image.fromarray(transform_array(img, self.masked, self.op)))


def embed_variants(cfg, wp, ids, crop_path, masked_path):
    import timm
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    todo = []
    for v in VARIANTS:
        p = wp["features"] / f"{FEAT[v]}.npz"
        if p.exists() and set(ids) <= set(E.load_feats(p)):
            log.info("%s: cached (%s)", v, p.name)
        else:
            todo.append(v)
    if not todo:
        return
    mcfg = cfg["megadescriptor"]
    tf = T.Compose([T.Resize((mcfg["image_size"], mcfg["image_size"])), T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(mcfg["model"], num_classes=0, pretrained=True).eval().to(device)
    use_amp = bool(mcfg.get("fp16")) and device == "cuda"
    for v in todo:
        masked, op = VARIANTS[v]
        paths = [(masked_path if masked else crop_path)[i] for i in ids]
        loader = DataLoader(Variant(paths, masked, op, tf), batch_size=mcfg["batch_size"],
                            num_workers=mcfg["num_workers"], pin_memory=device == "cuda")
        feats = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            for x in tqdm(loader, desc=f"MegaDescriptor ({v})"):
                feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        out = wp["features"] / f"{FEAT[v]}.npz"
        np.savez_compressed(out, image_ids=np.asarray(ids, dtype=object), feats=np.concatenate(feats))
        log.info("Saved %d features -> %s", len(ids), out)


def example_sheet(df, crop_path, masked_path, path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = df.sample(frac=1.0, random_state=0).drop_duplicates("zoo").head(n)
    cols = ["original", "grey", "cnorm", "masked", "masked_grey", "masked_cnorm"]
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(len(cols) * 2.0, len(pick) * 1.1))
    for r, iid in enumerate(pick["image_id"]):
        orig = np.asarray(Image.open(crop_path[iid]).convert("RGB"))
        msk = np.asarray(Image.open(masked_path[iid]).convert("RGB"))
        ims = [orig, transform_array(orig, False, "grey"), transform_array(orig, False, "cnorm"),
               msk, transform_array(msk, True, "grey"), transform_array(msk, True, "cnorm")]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
        axes[r, 0].text(-5, im.shape[0] / 2, pick["zoo"].iloc[r], fontsize=6, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Example sheet -> %s", path)


def evaluate(name, df, S, reps, seed, units):
    ident, seq, zoo = (df[c].to_numpy().astype(str) for c in ("identity", "sequence", "zoo"))
    out = {"subset": name, "n_images": len(df), "n_sequences": int(df["sequence"].nunique()),
           "n_bears": int(df["identity"].nunique()), "negatives": {}, "gap": {}}
    boots = {}
    for mode in ("all", "within_zoo"):
        i, j, y = M.make_pairs(ident, seq, zoo, within_zoo=(mode == "within_zoo"))
        sc = {m: s[i, j] for m, s in S.items()}
        r = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
             "auroc": {m: float(roc_auc_score(y, s)) for m, s in sc.items()}, "bootstrap": {}}
        for unit in units:
            codes, u = pd.factorize(df[unit])
            boot = M.paired_bootstrap(sc, y, codes[i], codes[j], len(u), reps, seed)
            boots[(mode, unit)] = boot
            b = {"ci95": {m: list(M.ci(v)) for m, v in boot.items()}, "deltas": {}}
            for a, c in DELTAS:
                if a in sc and c in sc:
                    d = boot[a] - boot[c]
                    b["deltas"][f"{a}|{c}"] = {"point": r["auroc"][a] - r["auroc"][c], "ci95": list(M.ci(d)),
                                               "p_bootstrap": M.bootstrap_p(d)}
            r["bootstrap"][unit] = b
        out["negatives"][mode] = r
        log.info("[%s | %s] %s", name, mode, "  ".join(f"{NAMES[m]}: {v:.3f}" for m, v in r["auroc"].items()))
    # gap = AUROC(all) - AUROC(within-zoo): how much the model gains from cross-zoo negatives.
    # Same seed + same units in both modes -> identical draws, so gaps and gap changes are paired.
    A, W = out["negatives"]["all"]["auroc"], out["negatives"]["within_zoo"]["auroc"]
    for unit in units:
        ga = {m: boots[("all", unit)][m] - boots[("within_zoo", unit)][m] for m in S}
        g = {"gap": {m: {"point": A[m] - W[m], "ci95": list(M.ci(ga[m]))} for m in S}, "change": {}}
        for a, c in DELTAS:
            if a in S and c in S:
                d = ga[a] - ga[c]
                g["change"][f"{a}|{c}"] = {"point": (A[a] - W[a]) - (A[c] - W[c]), "ci95": list(M.ci(d)),
                                           "p_bootstrap": M.bootstrap_p(d)}
        out["gap"][unit] = g
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
    mdc = read_csv(wp["md_csv"], "megadetector.csv (stage 01)")
    crop_path = mdc.set_index("image_id")["crop_path"].to_dict()
    masked_path = masks.set_index("image_id")["masked_path"].to_dict()
    base = {m: E.load_feats(wp["features"] / f"{m}.npz") for m in (MD, MDM, BD)}

    # identical frame selection to stage 7
    df = sample.merge(faces, on="image_id", how="left").merge(masks[["image_id", "mask_frac", "mask_flag"]],
                                                              on="image_id", how="left")
    df["zoo"] = df["zoo"].fillna("unknown").astype(str)
    df["identity"] = df["identity"].astype(str)
    has = {m: df["image_id"].isin(base[m].keys()) for m in base}
    w = int(bcfg["primary_min_face_width_px"])
    prim = df["landmarks_ok"].fillna(False).astype(bool) & (df["face_w_px"] >= w) & has[BD] & has[MD]
    sub, _ = E.select_frames(df[prim], k, seed)
    sub_all, _ = E.select_frames(df[has[MD]], k, seed)
    ids = sorted(set(sub["image_id"]) | set(sub_all["image_id"]))
    log.info("Embedding %d frames x %d variants", len(ids), len(VARIANTS))

    example_sheet(sub, crop_path, masked_path, res_dir / "colour_examples.png")
    embed_variants(cfg, wp, ids, crop_path, masked_path)
    feats = dict(base, **{FEAT[v]: E.load_feats(wp["features"] / f"{FEAT[v]}.npz") for v in VARIANTS})

    def scores(d, models):
        return {m: M.similarity(np.stack([feats[m][i] for i in d["image_id"]]), SIM[m]) for m in models}

    out = {"target_std": TARGET_STD}
    out["primary"] = evaluate(f"face_w>={w}px", sub, scores(sub, list(NAMES)), reps, seed, ("sequence", "identity"))
    md_models = [m for m in NAMES if m != BD]
    out["all_frames"] = evaluate("all_frames", sub_all, scores(sub_all, md_models), reps, seed, ("sequence",))
    (res_dir / "colour.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    write_md(out, res_dir / "colour.md", reps)
    log.info("Done. See %s", res_dir / "colour.md")


def write_md(out, path, reps):
    L = ["# Colour removal (stage 8)", "",
         "MegaDescriptor re-embedded on greyscale and grey-world colour-normalised crops (channel means -> 128, "
         f"shared scale -> std {out['target_std']:.0f}), with and without the SAM background mask. "
         "Gap = AUROC(all negatives) − AUROC(within-zoo negatives). Designed after the stage-7 results; secondary.", ""]
    for key in ("primary", "all_frames"):
        r = out[key]
        models = list(r["negatives"]["all"]["auroc"])
        L += [f"## {r['subset']} ({r['n_images']} images, {r['n_sequences']} sequences, {r['n_bears']} bears)", "",
              "| Model | All negatives | Within-zoo | Gap [95% CI, sequence] | Rank-1 | mAP |", "|---|---|---|---|---|---|"]
        for m in models:
            a, wz = r["negatives"]["all"], r["negatives"]["within_zoo"]
            ca, cw = a["bootstrap"]["sequence"]["ci95"][m], wz["bootstrap"]["sequence"]["ci95"][m]
            g = r["gap"]["sequence"]["gap"][m]
            L.append(f"| {NAMES[m]} | {a['auroc'][m]:.3f} [{ca[0]:.3f}, {ca[1]:.3f}] | {wz['auroc'][m]:.3f} "
                     f"[{cw[0]:.3f}, {cw[1]:.3f}] | {g['point']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] | "
                     f"{r['ranking'][m]['rank1']:.3f} | {r['ranking'][m]['mAP']:.3f} |")
        L += ["", "| Comparison | Unit | Δ all [95% CI] | p | Δ within-zoo [95% CI] | p | Δ gap [95% CI] | p |",
              "|---|---|---|---|---|---|---|---|"]
        for unit in r["gap"]:
            for kk, gc in r["gap"][unit]["change"].items():
                a, c = kk.split("|")
                da = r["negatives"]["all"]["bootstrap"][unit]["deltas"][kk]
                dw = r["negatives"]["within_zoo"]["bootstrap"][unit]["deltas"][kk]
                f = lambda d: f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {E.fmt_p(d['p_bootstrap'], reps)}"
                L.append(f"| {NAMES[a]} − {NAMES[c]} | {'bear' if unit == 'identity' else unit} | {f(da)} | {f(dw)} | {f(gc)} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
