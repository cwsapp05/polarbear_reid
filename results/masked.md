# Background masking (stage 7)

SAM 2.1-B masks, prompted with the MegaDetector box; background set to mid-grey. 13918 crops, median mask = 68% of the box, 17 flagged (mask under 5% or over 98% of the box). Designed after the stage-4 results; secondary.

## face_w>=24px (885 images, 386 sequences, 13 bears)

| Negatives | MegaDescriptor | MegaDescriptor, masked | BearID | Fused (MD + BearID) | Fused (masked MD + BearID) |
|---|---|---|---|---|---|
| all | 0.768 [0.747, 0.790] | 0.736 [0.710, 0.762] | 0.660 [0.639, 0.680] | 0.781 [0.762, 0.800] | 0.760 [0.739, 0.783] |
| within-zoo | 0.622 [0.592, 0.655] | 0.586 [0.555, 0.617] | 0.618 [0.589, 0.644] | 0.664 [0.634, 0.692] | 0.638 [0.608, 0.665] |

| Negatives | Comparison | Unit | Δ [95% CI] | p |
|---|---|---|---|---|
| all | MegaDescriptor, masked − MegaDescriptor | sequence | -0.032 [-0.050, -0.013] | <0.001 |
| all | BearID − MegaDescriptor, masked | sequence | -0.076 [-0.110, -0.047] | <0.001 |
| all | BearID − MegaDescriptor | sequence | -0.108 [-0.139, -0.081] | <0.001 |
| all | Fused (masked MD + BearID) − MegaDescriptor, masked | sequence | +0.024 [+0.008, +0.038] | <0.001 |
| all | Fused (masked MD + BearID) − Fused (MD + BearID) | sequence | -0.021 [-0.033, -0.010] | <0.001 |
| all | MegaDescriptor, masked − MegaDescriptor | bear | -0.032 [-0.078, +0.005] | 0.096 |
| all | BearID − MegaDescriptor, masked | bear | -0.076 [-0.142, -0.008] | 0.030 |
| all | BearID − MegaDescriptor | bear | -0.108 [-0.169, -0.060] | <0.001 |
| all | Fused (masked MD + BearID) − MegaDescriptor, masked | bear | +0.024 [-0.006, +0.049] | 0.092 |
| all | Fused (masked MD + BearID) − Fused (MD + BearID) | bear | -0.021 [-0.051, +0.005] | 0.106 |
| within-zoo | MegaDescriptor, masked − MegaDescriptor | sequence | -0.036 [-0.063, -0.013] | <0.001 |
| within-zoo | BearID − MegaDescriptor, masked | sequence | +0.032 [-0.014, +0.072] | 0.168 |
| within-zoo | BearID − MegaDescriptor | sequence | -0.004 [-0.047, +0.036] | 0.794 |
| within-zoo | Fused (masked MD + BearID) − MegaDescriptor, masked | sequence | +0.051 [+0.030, +0.071] | <0.001 |
| within-zoo | Fused (masked MD + BearID) − Fused (MD + BearID) | sequence | -0.026 [-0.043, -0.010] | <0.001 |
| within-zoo | MegaDescriptor, masked − MegaDescriptor | bear | -0.036 [-0.086, +0.027] | 0.297 |
| within-zoo | BearID − MegaDescriptor, masked | bear | +0.032 [-0.198, +0.165] | 0.954 |
| within-zoo | BearID − MegaDescriptor | bear | -0.004 [-0.232, +0.108] | 0.888 |
| within-zoo | Fused (masked MD + BearID) − MegaDescriptor, masked | bear | +0.051 [-0.051, +0.115] | 0.532 |
| within-zoo | Fused (masked MD + BearID) − Fused (MD + BearID) | bear | -0.026 [-0.059, +0.020] | 0.279 |

| Model | Rank-1 | mAP |
|---|---|---|
| MegaDescriptor | 0.812 | 0.442 |
| MegaDescriptor, masked | 0.806 | 0.425 |
| BearID | 0.632 | 0.298 |
| Fused (MD + BearID) | 0.863 | 0.468 |
| Fused (masked MD + BearID) | 0.832 | 0.452 |

## all_frames (4293 images, 1431 sequences, 13 bears)

| Negatives | MegaDescriptor | MegaDescriptor, masked |
|---|---|---|
| all | 0.674 [0.662, 0.685] | 0.661 [0.650, 0.673] |
| within-zoo | 0.605 [0.594, 0.616] | 0.582 [0.573, 0.591] |

| Negatives | Comparison | Unit | Δ [95% CI] | p |
|---|---|---|---|---|
| all | MegaDescriptor, masked − MegaDescriptor | sequence | -0.012 [-0.019, -0.004] | 0.002 |
| within-zoo | MegaDescriptor, masked − MegaDescriptor | sequence | -0.023 [-0.032, -0.014] | <0.001 |

| Model | Rank-1 | mAP |
|---|---|---|
| MegaDescriptor | 0.828 | 0.260 |
| MegaDescriptor, masked | 0.780 | 0.241 |
