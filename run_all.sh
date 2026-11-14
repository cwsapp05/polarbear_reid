#!/usr/bin/env bash
# Full pipeline. Each stage writes its outputs to work/ and can be re-run on its own.
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-config.yaml}"

python check_megadescriptor_training_data.py
python 00_prepare_data.py      --config "$CFG"
python 01_megadetector.py      --config "$CFG"
python 02_face_filter.py       --config "$CFG"
echo ">>> Check results/face_width_hist.png. Every threshold in min_face_widths_px is scored;"
echo ">>> primary_min_face_width_px only picks the headline row. Change it and re-run 04_evaluate.py if needed."
python 03_extract_features.py  --config "$CFG"
python 04_evaluate.py          --config "$CFG"
python 05_robustness.py        --config "$CFG"
python 06_mask_background.py   --config "$CFG"
python 07_masked_eval.py       --config "$CFG"
python 08_colour_eval.py        --config "$CFG"
