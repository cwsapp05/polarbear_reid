# PolarBearVidID: MegaDescriptor (body) vs BearID (face)

Verification AUROC, pairs from different sequences only, 95% paired cluster-bootstrap CIs.

*Fused* = equal-weight mean of the two models' z-scored similarities (untuned, label-free normalisation). It was added after the first results showed the models' errors overlap little, so treat it as a secondary analysis.

## Negatives: all

| Subset | Images | Seq. | Bears | MegaDescriptor | BearID | Δ (BearID − MD) [95% CI] | p | Fused | Δ (Fused − MD) [95% CI] | p |
|---|---|---|---|---|---|---|---|---|---|---|
| face_w>=0px | 888 | 386 | 13 | 0.770 [0.747, 0.793] | 0.659 [0.637, 0.681] | -0.110 [-0.140, -0.081] | <0.001 | 0.781 [0.760, 0.801] | +0.011 [-0.003, +0.026] | 0.126 |
| face_w>=16px | 888 | 386 | 13 | 0.773 [0.753, 0.796] | 0.662 [0.640, 0.684] | -0.111 [-0.140, -0.085] | <0.001 | 0.785 [0.766, 0.806] | +0.012 [-0.002, +0.024] | 0.110 |
| face_w>=24px | 885 | 386 | 13 | 0.768 [0.747, 0.790] | 0.660 [0.639, 0.680] | -0.108 [-0.139, -0.081] | <0.001 | 0.781 [0.762, 0.800] | +0.014 [-0.001, +0.028] | 0.064 |
| face_w>=32px | 689 | 325 | 13 | 0.783 [0.759, 0.808] | 0.676 [0.650, 0.703] | -0.107 [-0.138, -0.077] | <0.001 | 0.796 [0.772, 0.821] | +0.013 [-0.002, +0.028] | 0.076 |
| all_frames | 4293 | 1431 | 13 | 0.674 [0.662, 0.685] | – | – | – | – | – | – |

## Negatives: within-zoo

| Subset | Images | Seq. | Bears | MegaDescriptor | BearID | Δ (BearID − MD) [95% CI] | p | Fused | Δ (Fused − MD) [95% CI] | p |
|---|---|---|---|---|---|---|---|---|---|---|
| face_w>=0px | 888 | 386 | 13 | 0.624 [0.590, 0.657] | 0.616 [0.588, 0.645] | -0.008 [-0.048, +0.035] | 0.694 | 0.661 [0.630, 0.691] | +0.037 [+0.017, +0.057] | <0.001 |
| face_w>=16px | 888 | 386 | 13 | 0.628 [0.595, 0.662] | 0.623 [0.597, 0.651] | -0.004 [-0.047, +0.037] | 0.824 | 0.669 [0.641, 0.698] | +0.041 [+0.020, +0.062] | <0.001 |
| face_w>=24px | 885 | 386 | 13 | 0.622 [0.592, 0.655] | 0.618 [0.589, 0.644] | -0.004 [-0.047, +0.036] | 0.794 | 0.664 [0.634, 0.692] | +0.041 [+0.020, +0.061] | <0.001 |
| face_w>=32px | 689 | 325 | 13 | 0.627 [0.593, 0.664] | 0.623 [0.591, 0.659] | -0.004 [-0.051, +0.040] | 0.874 | 0.668 [0.633, 0.704] | +0.041 [+0.018, +0.064] | <0.001 |
| all_frames | 4293 | 1431 | 13 | 0.605 [0.594, 0.616] | – | – | – | – | – | – |

## Headline (face_w>=24px)

- all negatives: MegaDescriptor 0.768 [0.747, 0.790], BearID 0.660 [0.639, 0.680]; Δ = -0.108 [-0.139, -0.081], p < 0.001 - MegaDescriptor scores higher, and the interval excludes zero.
- within-zoo negatives: MegaDescriptor 0.622 [0.592, 0.655], BearID 0.618 [0.589, 0.644]; Δ = -0.004 [-0.047, +0.036], p = 0.794 - the difference is not distinguishable from zero at this sample size.

Secondary (fused score):
- all negatives: fused 0.781 [0.762, 0.800]; Δ vs MegaDescriptor = +0.014 [-0.001, +0.028], p 0.064; Δ vs BearID = +0.122 [+0.106, +0.139], p <0.001.
- within-zoo negatives: fused 0.664 [0.634, 0.692]; Δ vs MegaDescriptor = +0.041 [+0.020, +0.061], p <0.001; Δ vs BearID = +0.045 [+0.022, +0.071], p <0.001.

## Retrieval (leave-one-sequence-out)

| Subset | Model | Rank-1 | mAP |
|---|---|---|---|
| face_w>=0px | megadescriptor | 0.814 | 0.447 |
| face_w>=0px | bearid | 0.643 | 0.303 |
| face_w>=0px | fused | 0.865 | 0.473 |
| face_w>=16px | megadescriptor | 0.825 | 0.449 |
| face_w>=16px | bearid | 0.657 | 0.301 |
| face_w>=16px | fused | 0.870 | 0.474 |
| face_w>=24px | megadescriptor | 0.812 | 0.442 |
| face_w>=24px | bearid | 0.632 | 0.298 |
| face_w>=24px | fused | 0.863 | 0.468 |
| face_w>=32px | megadescriptor | 0.817 | 0.457 |
| face_w>=32px | bearid | 0.637 | 0.326 |
| face_w>=32px | fused | 0.842 | 0.486 |
| all_frames | megadescriptor | 0.828 | 0.260 |

The fused score only exists on face-visible frames (see the stage 2 log for how many); elsewhere a hybrid falls back to MegaDescriptor alone. Caveats: 13 zoo bears from 6 institutions; both models are used zero-shot. BearID's networks were trained on brown bears, so this measures transfer to polar bears, not BearID's ceiling.
