# Brown bear head crops: MegaDescriptor vs BearID

Same comparison as the PolarBearVidID experiment in the parent folder, on the McNeil River
brown bear re-ID release (Rosenberg, Zhou, ..., A. Mathis; 72,939 head crops, 109 bears, 2017-2022,
CC BY-NC 4.0). Both models see the dataset's own head crops, so this compares models, not body vs face.

```
python bb_00_sample.py           # Windows: catalogue, 2 photos per bear-day, padded BearID inputs
python bb_01_bearid.py           # WSL (~/pbreid-venv): bearface -> bearchip -> bearembed  (~50 min)
python bb_02_megadescriptor.py   # Windows GPU (~5 min)
python bb_03_evaluate.py         # AUROC same-season / cross-season, encounter + bear bootstrap (~1 h)
```

Choices made before scoring (from a 100-image smoke test of detection rates only):
- BearID's brown-bear face network (`bearface_network.dat`); crops resized to 384 px and padded 20% grey.
- Faces need eyes + nose inside the box (bearchip's alignment points); `head_top` falls outside the
  box on tight head crops and failed the all-six check on most frontal faces.

Results: `results/summary.md`. BearID beats MegaDescriptor by ~0.07 AUROC on the frames where it finds a
face (56%; the rest are mostly profiles), in both same-season and cross-season pairs.
