"""Stage 6: remove the background from every body crop and re-embed it with MegaDescriptor.

Tests whether MegaDescriptor's lead over BearID comes from the bear or from its surroundings
(enclosure, water, camera, light). Each MegaDetector crop is segmented with SAM, prompted with the
detector box (the crop minus its padding). Everything outside the bear mask is set to mid-grey
(128, which MegaDescriptor's normalisation maps to 0), and the masked crop is embedded exactly
like stage 3.

  python 06_mask_background.py --limit 50      # smoke test: masks only, plus the example sheet
  python 06_mask_background.py                 # mask all crops, then embed

Output: work/crops_masked/, work/masks.csv (mask_frac, mask_flag per frame),
        work/features/megadescriptor_masked.npz, results/mask_examples.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import base_argparser, get_logger, load_config, read_csv, work_paths

log = get_logger("mask")
GREY = 128
SAM_WEIGHTS = "sam2.1_b.pt"   # ultralytics downloads it on first use
MIN_FRAC, MAX_FRAC = 0.05, 0.98  # mask area outside this range (fraction of the prompt box) = flagged


class Crops:
    """Map-style dataset over image files; module-level so Windows DataLoader workers can pickle it."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def save(path, image_ids, feats):
    np.savez_compressed(path, image_ids=np.asarray(image_ids, dtype=object), feats=np.asarray(feats, np.float32))
    log.info("Saved %d x %d features -> %s", len(image_ids), np.asarray(feats).shape[-1], path)


def masked_path(wp, image_id):
    return wp["work"] / "crops_masked" / f"{image_id}.png"


def segment(model, img: np.ndarray, pad: float):
    """SAM mask for the animal in one crop. The box prompt is the crop minus MegaDetector's padding."""
    import cv2
    h, w = img.shape[:2]
    ix, iy = w * pad / (1 + 2 * pad), h * pad / (1 + 2 * pad)
    box = [ix, iy, w - ix, h - iy]
    res = model(img[:, :, ::-1].copy(), bboxes=[box], verbose=False)[0]  # ultralytics expects BGR arrays
    if res.masks is None or len(res.masks.data) == 0:
        return np.zeros((h, w), bool), box
    m = res.masks.data.cpu().numpy().astype(bool)
    m = m[np.argmax(m.reshape(len(m), -1).sum(1))]
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    # drop stray specks (rock, grass); keep components >= 20% of the largest so a bear split by a post stays whole
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        m = np.isin(lab, 1 + np.flatnonzero(areas >= 0.2 * areas.max()))
    # 1 px dilation keeps the fur outline that SAM tends to shave off
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return m, box


def make_masks(cfg, wp, limit):
    from ultralytics import SAM
    md = read_csv(wp["md_csv"], "megadetector.csv (stage 01)")
    md = md[md["md_ok"].astype(bool)].reset_index(drop=True)
    if limit:
        md = md.sample(n=min(limit, len(md)), random_state=0).reset_index(drop=True)
    pad = float(cfg["megadetector"]["pad_frac"])
    weights = cfg["paths"]["work_dir"].parent / "tools" / SAM_WEIGHTS
    model = SAM(str(weights) if weights.exists() else SAM_WEIGHTS)

    out_csv = wp["work"] / ("masks_smoke.csv" if limit else "masks.csv")
    done = pd.read_csv(out_csv) if out_csv.exists() and not limit else pd.DataFrame(columns=["image_id"])
    done_ids = set(done["image_id"])
    rows = done.to_dict(orient="records")
    todo = md[~md["image_id"].isin(done_ids)]
    log.info("Masking %d crops (%d already done)", len(todo), len(done_ids))
    for k, r in enumerate(tqdm(todo.itertuples(index=False), total=len(todo), desc="SAM")):
        img = np.asarray(Image.open(r.crop_path).convert("RGB"))
        m, (bx0, by0, bx1, by1) = segment(model, img, pad)
        frac = m.sum() / max((bx1 - bx0) * (by1 - by0), 1)
        out = np.where(m[..., None], img, GREY).astype(np.uint8)
        p = masked_path(wp, r.image_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(p)
        rows.append({"image_id": r.image_id, "masked_path": str(p), "mask_frac": round(float(frac), 4),
                     "mask_flag": bool(frac < MIN_FRAC or frac > MAX_FRAC)})
        if (k + 1) % 500 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info("mask_frac median %.2f (IQR %.2f-%.2f); flagged %d / %d (outside %.2f-%.2f of the box)",
             df["mask_frac"].median(), df["mask_frac"].quantile(.25), df["mask_frac"].quantile(.75),
             int(df["mask_flag"].sum()), len(df), MIN_FRAC, MAX_FRAC)
    example_sheet(md.set_index("image_id")["crop_path"], df, cfg["paths"]["results_dir"] / "mask_examples.png")
    return df


def example_sheet(crop_paths, df, path, n=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = pd.concat([df[df["mask_flag"]].head(4), df[~df["mask_flag"]].sample(n=min(n - 4, (~df["mask_flag"]).sum()),
                                                                                     random_state=1)]).head(n)
    cols = 6
    rws = int(np.ceil(len(pick) / cols)) * 2
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 2.2, rws * 1.5))
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(pick.itertuples(index=False)):
        a, b = axes[(k // cols) * 2, k % cols], axes[(k // cols) * 2 + 1, k % cols]
        a.imshow(Image.open(crop_paths[r.image_id]))
        b.imshow(Image.open(r.masked_path))
        b.set_title(f"{r.mask_frac:.2f}" + (" FLAG" if r.mask_flag else ""), fontsize=7)
    fig.suptitle("Original crop (upper) and SAM-masked crop (lower); number = mask area / box area", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Example sheet -> %s", path)


def embed(cfg, wp, masks):
    import timm
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    mcfg = cfg["megadescriptor"]
    tf = T.Compose([T.Resize((mcfg["image_size"], mcfg["image_size"])), T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(mcfg["model"], num_classes=0, pretrained=True).eval().to(device)
    loader = DataLoader(Crops(masks["masked_path"].tolist(), tf), batch_size=mcfg["batch_size"],
                        num_workers=mcfg["num_workers"], pin_memory=device == "cuda")
    feats = []
    use_amp = bool(mcfg.get("fp16")) and device == "cuda"
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        for x in tqdm(loader, desc="MegaDescriptor (masked)"):
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    save(wp["features"] / "megadescriptor_masked.npz", masks["image_id"], np.concatenate(feats))


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--limit", type=int, default=0, help="mask only N random crops (smoke test, no embedding)")
    ap.add_argument("--skip-masks", action="store_true", help="reuse work/masks.csv and only embed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp = work_paths(cfg)
    wp["work"] = cfg["paths"]["work_dir"]
    if args.skip_masks:
        masks = read_csv(wp["work"] / "masks.csv", "masks.csv (stage 06)")
    else:
        masks = make_masks(cfg, wp, args.limit)
    if not args.limit:
        embed(cfg, wp, masks)


if __name__ == "__main__":
    main()
