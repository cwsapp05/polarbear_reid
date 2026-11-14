# PolarBearVidID re-ID experiment: MegaDescriptor (body) vs BearID (face)

**Question.** On polar bear frames where the face is clearly visible, does a face-only model
(BearID) or a whole-body model (MegaDescriptor) do a better job of telling individuals apart?

**Pipeline.**

```
PolarBearVidID frames ─► 00 sample every k-th frame per sequence
                     ─► 01 MegaDetector: one animal crop per frame
                     ─► 02 BearID face detector + landmarks: "face visible" flag and face size
                     ─► 03 features: MegaDescriptor on the crop, BearID on the aligned face chip
                     ─► 04 pairwise verification AUROC, paired bootstrap, ROC plots
                     ─► 05 robustness: bear-level bootstrap, bear balance, held-out fusion weights
                     ─► 06/07 background masking (SAM) and re-scoring MegaDescriptor
                     ─► 08 greyscale / colour-normalised re-scoring of MegaDescriptor
```

Both models are scored on **exactly the same frames and pairs**, so the comparison is paired.

---

## 1. Setup (local GPU)

```bash
# Python 3.10+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt

# BearID (dlib C++ tools). Needs: build-essential cmake git libboost-filesystem-dev
#   libboost-system-dev libopenblas-dev liblapack-dev libjpeg-dev libpng-dev libx11-dev
# For GPU face detection, install CUDA + cuDNN first; dlib detects them automatically.
./setup_bearid.sh
```

`setup_bearid.sh` clones dlib 19.24.6, `bearid/bearid-dlib` and `hypraptive/bearid-models` into
`tools/` and builds `bearface`, `bearchip` and `bearembed`. (BearID's README says it was tested on
dlib 19.16; this pipeline was built and checked against 19.24.6 + Boost 1.83.)

**On Windows, build BearID inside WSL's own filesystem, not under `/mnt/c`.** Cloning onto the
Windows drive fails (`chmod on .../.git/config.lock failed: Operation not permitted`). What worked:
copy `setup_bearid.sh` to `~/pbreid-build/` in WSL (Ubuntu), run it there, then copy
`tools/bearid-dlib/build/{bearface,bearchip,bearembed}` and `tools/bearid-models/` back into this
folder's `tools/`. Stage 02 and the BearID half of stage 03 then run under WSL (the venv used was
`~/pbreid-venv`); `common.read_csv` maps the Windows paths in the CSVs to `/mnt/c/...`. Every other
stage ran on Windows.

## 2. Run

```bash
./run_all.sh                     # everything, using config.yaml
# or stage by stage:
python check_megadescriptor_training_data.py
python 00_prepare_data.py        # downloads 1.6 GB from Zenodo, checks MD5, builds the catalogue
python 01_megadetector.py        # add --limit 50 for a smoke test
python 02_face_filter.py
python 03_extract_features.py    # --model bearid | megadescriptor | all
python 04_evaluate.py
python 05_robustness.py          # secondary checks on the primary subset (see section 4)
python 06_mask_background.py     # SAM 2.1-B masks (~55 min on an RTX 2060) + re-embed; --limit 50 to smoke-test
python 07_masked_eval.py         # masked vs original MegaDescriptor on the same frames and pairs
python 08_colour_eval.py         # greyscale and colour-normalised MD (with/without mask); ~15 min + bootstraps
```

Stages 01 and 02 resume after interruptions. Stage 04 takes a few minutes and can be re-run on its own.

## 3. What each stage does, and why

| Stage | Output | Notes |
|---|---|---|
| `check_megadescriptor_training_data.py` | console | Checks the published MegaDescriptor training list for PolarBearVidID. **Result as of Nov 2026: not included** (29 datasets; the only bear species is IPanda50). That covers the T/S/B/L-224 and L-384 checkpoints. Newer variants (DINOv2-518, CLIP-336, T-CNN-288) may have been trained on WildlifeReID-10k, which *does* include PolarBearVidID, so don't swap them in without checking. |
| `00_prepare_data.py` | `work/catalogue.csv`, `work/sample.csv` | Sequences are about 100 frames at 12.5 fps, so neighbouring frames are near-duplicates. `frame_stride: 10` keeps about 13.8k frames. The script prints identity, sequence and image counts and compares them with the paper (13 / 1431 / 138,363). |
| `01_megadetector.py` | `work/megadetector.csv`, `work/crops/` | MDv5a, top animal box with 5% padding. Dataset frames are already crops, so a frame with no detection falls back to the full frame and is flagged (`md_fallback`). The log reports how much MegaDetector tightened the box. |
| `02_face_filter.py` | `work/faces.csv`, `results/face_width_hist.png`, `results/face_visibility_by_bear.csv` | Crops are upscaled to a 512 px long side, then BearID's `face_allbears.dat` detector (trained on all bear species) finds the face and 6 landmarks. A frame counts as **face visible** when a face is found, all landmarks fall inside the face box, and the face is at least *w* original pixels wide. Every *w* in `min_face_widths_px` is scored. |
| `03_extract_features.py` | `work/features/*.npz` | MegaDescriptor-L-384 on the body crop (model-card transform, 384×384). BearID: `bearchip` aligns each face on eyes and nose into a 150×150 chip, then `bearembed` produces a 128-d embedding. |
| `04_evaluate.py` | `results/summary.md`, `results_table.csv`, `results.json`, figures | See below. |
| `05_robustness.py` | `results/robustness.md`, `robustness.json` | Bear-level bootstrap, bear-balanced AUROC, leave-one-bear-out, leave-one-zoo-out fusion weights. See section 4. |
| `06_mask_background.py` | `work/crops_masked/`, `work/masks.csv`, `work/features/megadescriptor_masked.npz`, `results/mask_examples.png` | SAM 2.1-B (ultralytics; weights in `tools/`) prompted with the detector box; background set to grey 128; specks under 20% of the largest component dropped; 1 px dilation. |
| `07_masked_eval.py` | `results/masked.md`, `masked.json` | Original vs masked MegaDescriptor, BearID and both fusions on the stage-4 primary subset and all frames. |
| `08_colour_eval.py` | `work/features/megadescriptor_{grey,cnorm,masked_grey,masked_cnorm}.npz`, `results/colour.md`, `colour.json`, `colour_examples.png` | Greyscale (luminance) and grey-world colour normalisation (channel means to 128, shared scale to std 50), on original and masked crops; embeds only the evaluated frames; reports the all − within-zoo gap with a paired CI. |

### Evaluation design (stage 4)

- **Frames:** up to 3 face-visible frames per sequence, chosen at random with a fixed seed. Bears with fewer than 2 face-visible sequences are dropped, since they can't form a positive pair. Dropped bears are logged.
- **Pairs:** every two frames **from different sequences**. Same-sequence pairs are excluded because they are near-identical frames and would inflate every model's score.
- **Labels:** same bear = positive, different bears = negative.
- **Scores:** each model's native metric. Cosine similarity for MegaDescriptor; negative Euclidean distance for BearID, whose dlib metric-learning network is trained on Euclidean distance. Change this under `evaluation.similarity`.
- **Two sets of negatives:**
  - *all*: every different-bear pair.
  - *within-zoo*: only bears from the same zoo, so enclosure, camera and lighting cues can't help. If MegaDescriptor's lead shrinks a lot under within-zoo, part of its score came from context, not the bear.
- **Uncertainty:** a paired cluster bootstrap over sequences (1,000 replicates). Each replicate resamples sequences, and both models are scored on the same resample. That gives a CI for each AUROC and for **Δ = BearID − MegaDescriptor**, plus a two-sided bootstrap p-value. Pairs share images, so a standard DeLong test would understate the uncertainty. Set `bootstrap_unit: individual` for the stricter bear-level version; with 13 bears the intervals will be wide.
- **Also reported:** rank-1 accuracy and mAP, querying each frame against every other sequence, plus MegaDescriptor on **all** frames with no face requirement (`all_frames`).

### Picking the headline threshold

`primary_min_face_width_px` (default 24) only selects which subset gets the headline sentence,
the ROC curves and the similarity histograms. All thresholds are computed every time. Set it
from `face_width_hist.png` after stage 2, **before** looking at any AUROC. Choosing it after
seeing the results is a forking-paths problem.

## 4. Reading the results

`results/summary.md` contains tables for both negative sets, a headline paragraph for the
primary subset that says whether the Δ interval excludes zero, and the retrieval metrics.
`auroc_by_face_width.png` shows whether the ranking flips as faces get larger. That is the most
interesting plot for the face-vs-body question.

With `evaluation.fusion.enabled`, every face-visible subset also gets a **fused** score: the
equal-weight mean of the two models' similarities after z-scoring each over all frame pairs of
the subset (no labels, no tuning). It gets the same bootstrap, with Δ against each model. It was
added after the first results showed the two models' errors overlap little (rank-1 misses
correlate at about 0.2), so report it as secondary.

Check `face_visibility_by_bear.csv` too. If a few bears rarely face the camera, the face-visible
subset over-represents the others, so report that next to any conclusion.

### Stage 5 robustness checks (`results/robustness.md`, `robustness.json`)

`05_robustness.py` re-selects exactly the stage-4 primary subset (same seed) and adds four checks.
All were designed after seeing the stage-4 results, so they are secondary:

- **A. Composition.** Images, sequences and share of positive pairs per bear.
- **B. Bootstrap unit.** Sequence-level and bear-level cluster bootstrap side by side. Only the
  bear-level interval speaks to *new bears*; the sequence-level one covers new footage of these 13.
- **C. Bear balance.** AUROC with each bear's positives and each bear pair's negatives weighted
  equally, plus leave-one-bear-out deltas. Bears with very few frames get as much weight as the
  best-covered ones here, so the balanced numbers are noisy.
- **D. Tuned fusion.** `w·z(MD) + (1−w)·z(BearID)`, with `w` picked on a 0.05 grid on five zoos and
  applied to the sixth (leave-one-zoo-out). Cross-zoo pairs use the mean of their two zoos'
  held-out weights. The bootstrap is conditional on the chosen weights.

## 5. Caveats

- **Domain shift for BearID.** Its embedding network was trained on brown bears. This experiment measures *zero-shot transfer* to polar bears, not what a face model could do with polar bear training data. Polar bears also lack the distinctive coat markings MegaDescriptor relies on in other species, so neither model is in its home domain.
- **Resolution.** The frames are 256×128, so faces are small, and the face-width sweep is how this pipeline addresses it.
- **Sequence parsing.** File name → sequence follows the `wildlife-datasets` loader (characters 7–10 of the file name). If stage 0's sequence count is far from 1431, set `dataset.sequence_regex`. This matters because same-sequence pairs are excluded.
- **Small population.** 13 captive bears in 6 zoos, which limits how far the result generalises to wild populations.

- **Unbalanced face visibility.** Only 1,272 of 13.9k sampled frames have a usable face, and
  Vera, Larissa and Tonja supply 77% of the positive pairs in the primary subset. Lloyd,
  Charlotte and Sprinter have 7–11 face-visible frames each.

## 6. Verification

While building:

- BearID tools built from source (dlib 19.24.6, Boost 1.83, CPU). The wrapper drove `bearface` → `bearchip` → `bearembed` end to end, and every embedding mapped back to its source frame.
- The tie-aware weighted AUROC matches `sklearn.roc_auc_score` exactly under uniform weights, ties included. The bootstrap and rank metrics behaved as expected on synthetic embeddings with known signal.
- The full pipeline ran on a synthetic stand-in dataset first.

On the real data (Nov 11–13, 2026), every stage has now run end to end:

- Stages 00–01 (Windows): PolarBearVidID downloaded from Zenodo; about 13.9k frames sampled and cropped with MDv5a.
- Stage 02 (WSL): BearID's `face_allbears.dat` found faces on real polar bear frames; 1,272 frames pass the landmark check (`face_visibility_by_bear.csv`).
- Stage 03: MegaDescriptor-L-384 loaded from Hugging Face and embedded 13,918 crops (Windows); BearID produced 1,284 chips and 128-d embeddings (WSL).
- Stage 04: four face-width subsets plus all frames, 1,000 bootstrap replicates, with fusion.
- Stage 05: reproduces stage 4's sequence-bootstrap numbers exactly, which confirms it uses the same frames.

## 7. Results so far (primary subset, face width ≥ 24 px; 885 images, 386 sequences, 13 bears)

| Negatives | MegaDescriptor | BearID | Fused (equal) | Δ BearID − MD, bear-level 95% CI |
|---|---|---|---|---|
| All | 0.768 | 0.660 | 0.781 | −0.108 [−0.169, −0.060] |
| Within-zoo | 0.622 | 0.618 | 0.664 | −0.004 [−0.232, +0.108] |

- With all negatives, the body model clearly wins, even under the bear-level bootstrap.
- With within-zoo negatives the two models tie, so much of MegaDescriptor's lead is probably context (enclosure, camera, light).
- Equal-weight fusion improves within-zoo AUROC by +0.041. That holds under the sequence-level bootstrap only (p < 0.001); at the bear level the CI is [−0.057, +0.099].
- Tuning the fusion weight leave-one-zoo-out gave no gain over equal weights. The chosen weights were 0.4–0.6 on MD within-zoo (0.65–0.70 with all negatives), and the out-of-fold within-zoo AUROC was 0.652 against 0.664 for equal weights.
- Weighting all bears equally shrinks BearID's all-negatives deficit to −0.075 but widens its within-zoo deficit to −0.070. The fusion gain within zoos disappears under that weighting (+0.000), so it comes mainly from the well-covered bears.
- **Background masking (stages 6–7):** masking cost MegaDescriptor 0.032 AUROC with all negatives and 0.036 within zoos — the same. Masked MD still beats BearID with all negatives (Δ −0.076, bear-level p = 0.03), so the cross-zoo advantage is not the enclosure in the frame. Within zoos BearID is nominally ahead of masked MD (+0.032, n.s.).
- **Colour removal (stage 8):** grey-world colour normalisation slightly *helps* MD (+0.012 all, +0.002 within-zoo) and does not shrink the all − within-zoo gap (0.155 vs 0.145). Greyscale shrinks the gap by 0.031 on face-visible frames (sequence p = 0.01, bear-level p = 0.8) and not at all on all frames. Masked + colour-normalised MD keeps a gap of 0.141 and still beats BearID with all negatives (Δ −0.078, bear-level p = 0.03). So colour cast, exposure and contrast do not explain the cross-zoo lead; image quality (resolution/blur per camera) and relatedness remain untested.
