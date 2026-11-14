"""Check whether the MegaDescriptor training set included PolarBearVidID (train/test leakage).

The WildlifeDatasets team publishes the image list used to train the original MegaDescriptor
models (T/S/B/L-224 and L-384). This script downloads it and lists the datasets inside.
Newer variants (e.g. MegaDescriptor-DINOv2-518, -CLIP-336, -T-CNN-288) may have been trained on
WildlifeReID-10k, which DOES contain a PolarBearVidID subsample - check their model cards before
using them here.
"""
import io
import sys
import urllib.request

import pandas as pd

URL = ("https://raw.githubusercontent.com/WildlifeDatasets/wildlife-tools/main/"
       "baselines/data/metadata/combined/combined_all.csv")


def main() -> None:
    print(f"Fetching {URL}")
    with urllib.request.urlopen(URL) as r:
        df = pd.read_csv(io.BytesIO(r.read()), usecols=["dataset", "split"])
    counts = df[df["split"] == "train"]["dataset"].value_counts()
    print(f"\n{len(counts)} datasets in the published MegaDescriptor training split:")
    print(counts.to_string())
    leak = [d for d in counts.index if "polar" in d.lower()]
    bears = [d for d in counts.index if any(s in d.lower() for s in ("bear", "panda"))]
    print()
    if leak:
        print(f"LEAKAGE: {leak} is in the training data. MegaDescriptor results on PolarBearVidID are inflated.")
        sys.exit(1)
    print("OK: PolarBearVidID is not in the published training split.")
    if bears:
        print(f"Note: other bear datasets in training: {bears} (related species, not leakage).")


if __name__ == "__main__":
    main()
