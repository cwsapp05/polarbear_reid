# Brown bears (McNeil River heads): MegaDescriptor vs BearID

3939 sampled head crops; BearID found a usable face (eyes + nose) in 2203 (56%). Pairs within one encounter (bear-day) are never scored.

## bearid_ok (2192 images, 1460 encounters, 94 bears)

| Pairs | Positives | Negatives | MegaDescriptor AUROC [95% CI, encounter] | BearID AUROC [95% CI, encounter] | Fused AUROC [95% CI, encounter] |
|---|---|---|---|---|---|
| all | 38,020 | 2,362,584 | 0.603 [0.591, 0.616] | 0.675 [0.662, 0.689] | 0.678 [0.665, 0.691] |
| same-year | 7,859 | 405,884 | 0.623 [0.602, 0.644] | 0.707 [0.688, 0.726] | 0.706 [0.686, 0.725] |
| cross-year | 30,161 | 1,956,700 | 0.598 [0.584, 0.611] | 0.667 [0.653, 0.682] | 0.671 [0.658, 0.684] |

| Pairs | Comparison | Unit | Δ [95% CI] | p |
|---|---|---|---|---|
| all | BearID − MegaDescriptor | encounter | +0.072 [+0.055, +0.089] | <0.001 |
| all | Fused − MegaDescriptor | encounter | +0.075 [+0.066, +0.084] | <0.001 |
| all | Fused − BearID | encounter | +0.003 [-0.006, +0.012] | 0.550 |
| all | BearID − MegaDescriptor | bear | +0.072 [+0.039, +0.104] | <0.001 |
| all | Fused − MegaDescriptor | bear | +0.075 [+0.059, +0.090] | <0.001 |
| all | Fused − BearID | bear | +0.003 [-0.015, +0.023] | 0.752 |
| same-year | BearID − MegaDescriptor | encounter | +0.084 [+0.060, +0.107] | <0.001 |
| same-year | Fused − MegaDescriptor | encounter | +0.083 [+0.070, +0.095] | <0.001 |
| same-year | Fused − BearID | encounter | -0.001 [-0.015, +0.013] | 0.874 |
| same-year | BearID − MegaDescriptor | bear | +0.084 [+0.048, +0.120] | <0.001 |
| same-year | Fused − MegaDescriptor | bear | +0.083 [+0.065, +0.100] | <0.001 |
| same-year | Fused − BearID | bear | -0.001 [-0.021, +0.018] | 0.986 |
| cross-year | BearID − MegaDescriptor | encounter | +0.069 [+0.051, +0.088] | <0.001 |
| cross-year | Fused − MegaDescriptor | encounter | +0.073 [+0.063, +0.083] | <0.001 |
| cross-year | Fused − BearID | encounter | +0.004 [-0.006, +0.013] | 0.478 |
| cross-year | BearID − MegaDescriptor | bear | +0.069 [+0.033, +0.102] | <0.001 |
| cross-year | Fused − MegaDescriptor | bear | +0.073 [+0.057, +0.090] | <0.001 |
| cross-year | Fused − BearID | bear | +0.004 [-0.016, +0.026] | 0.672 |

| Model | Rank-1 (other days) | mAP (other days) | Rank-1 (other seasons) | mAP (other seasons) |
|---|---|---|---|---|
| MegaDescriptor | 0.142 | 0.048 | 0.101 | 0.043 |
| BearID | 0.160 | 0.064 | 0.104 | 0.054 |
| Fused | 0.248 | 0.078 | 0.161 | 0.065 |

## all_frames (3939 images, 1991 encounters, 101 bears)

| Pairs | Positives | Negatives | MegaDescriptor AUROC [95% CI, encounter] |
|---|---|---|---|
| all | 119,181 | 7,634,762 | 0.580 [0.571, 0.589] |
| same-year | 24,334 | 1,309,387 | 0.597 [0.585, 0.610] |
| cross-year | 94,847 | 6,325,375 | 0.576 [0.566, 0.586] |

| Model | Rank-1 (other days) | mAP (other days) | Rank-1 (other seasons) | mAP (other seasons) |
|---|---|---|---|---|
| MegaDescriptor | 0.137 | 0.037 | 0.102 | 0.034 |
