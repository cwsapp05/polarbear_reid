"""Brown bear stage 5: MegaDescriptor embeddings for the whole catalogue (training data for step 3).

Step 3 of the ladder trains a small head on frozen MegaDescriptor features using each experiment's full
train_iid split, as PoseSwin was trained on the full split. Resumable in chunks.

Output: work/features/megadescriptor_catalogue.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import md_embed  # noqa: E402
from common import base_argparser, get_logger, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
log = get_logger("bb.mdcat")
CHUNK = 5000


def main() -> None:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config or HERE / "config.yaml")
    work = cfg["paths"]["work_dir"]
    cat = pd.read_csv(work / "catalogue.csv")
    part_dir = work / "features" / "md_catalogue_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(cat), CHUNK):
        p = part_dir / f"{k:06d}.npz"
        if p.exists():
            continue
        c = cat.iloc[k:k + CHUNK]
        md_embed.save(p, c["image_id"], md_embed.embed(c["path"], cfg["megadescriptor"], f"MD catalogue {k}"))
        log.info("chunk %d / %d done", k // CHUNK + 1, -(-len(cat) // CHUNK))
    parts = [np.load(p, allow_pickle=True) for p in sorted(part_dir.glob("*.npz"))]
    md_embed.save(work / "features" / "megadescriptor_catalogue.npz",
                  np.concatenate([z["image_ids"] for z in parts]), np.concatenate([z["feats"] for z in parts]))
    log.info("Saved %d catalogue embeddings", sum(len(z["image_ids"]) for z in parts))


if __name__ == "__main__":
    main()
