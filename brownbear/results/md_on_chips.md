# Brown bears: MegaDescriptor on head crops vs aligned face chips

3939 sampled head crops; BearID found a usable face (eyes + nose) in 2192 (56%). Pairs within one encounter (bear-day) are never scored.

## bearid_ok (2192 images, 1460 encounters, 94 bears)

| Pairs | Positives | Negatives | MD, head crop AUROC [95% CI, encounter] | MD, aligned face chip AUROC [95% CI, encounter] | BearID (face chip) AUROC [95% CI, encounter] |
|---|---|---|---|---|---|
| all | 38,020 | 2,362,584 | 0.603 [0.591, 0.616] | 0.593 [0.581, 0.606] | 0.675 [0.662, 0.689] |
| same-year | 7,859 | 405,884 | 0.623 [0.602, 0.644] | 0.609 [0.589, 0.630] | 0.707 [0.688, 0.726] |
| cross-year | 30,161 | 1,956,700 | 0.598 [0.584, 0.611] | 0.589 [0.576, 0.602] | 0.667 [0.653, 0.682] |

| Pairs | Comparison | Unit | Δ [95% CI] | p |
|---|---|---|---|---|
| all | MD, aligned face chip − MD, head crop | encounter | -0.010 [-0.018, -0.002] | 0.020 |
| all | BearID (face chip) − MD, aligned face chip | encounter | +0.082 [+0.065, +0.099] | <0.001 |
| all | BearID (face chip) − MD, head crop | encounter | +0.072 [+0.055, +0.089] | <0.001 |
| all | MD, aligned face chip − MD, head crop | bear | -0.010 [-0.025, +0.007] | 0.190 |
| all | BearID (face chip) − MD, aligned face chip | bear | +0.082 [+0.048, +0.120] | <0.001 |
| all | BearID (face chip) − MD, head crop | bear | +0.072 [+0.039, +0.104] | <0.001 |
| same-year | MD, aligned face chip − MD, head crop | encounter | -0.014 [-0.028, -0.000] | 0.046 |
| same-year | BearID (face chip) − MD, aligned face chip | encounter | +0.098 [+0.073, +0.122] | <0.001 |
| same-year | BearID (face chip) − MD, head crop | encounter | +0.084 [+0.060, +0.107] | <0.001 |
| same-year | MD, aligned face chip − MD, head crop | bear | -0.014 [-0.033, +0.006] | 0.140 |
| same-year | BearID (face chip) − MD, aligned face chip | bear | +0.098 [+0.062, +0.137] | <0.001 |
| same-year | BearID (face chip) − MD, head crop | bear | +0.084 [+0.048, +0.120] | <0.001 |
| cross-year | MD, aligned face chip − MD, head crop | encounter | -0.009 [-0.017, -0.000] | 0.042 |
| cross-year | BearID (face chip) − MD, aligned face chip | encounter | +0.078 [+0.060, +0.097] | <0.001 |
| cross-year | BearID (face chip) − MD, head crop | encounter | +0.069 [+0.051, +0.088] | <0.001 |
| cross-year | MD, aligned face chip − MD, head crop | bear | -0.009 [-0.023, +0.007] | 0.276 |
| cross-year | BearID (face chip) − MD, aligned face chip | bear | +0.078 [+0.042, +0.118] | <0.001 |
| cross-year | BearID (face chip) − MD, head crop | bear | +0.069 [+0.033, +0.102] | <0.001 |

| Model | Rank-1 (other days) | mAP (other days) | Rank-1 (other seasons) | mAP (other seasons) |
|---|---|---|---|---|
| MD, head crop | 0.142 | 0.048 | 0.101 | 0.043 |
| MD, aligned face chip | 0.134 | 0.044 | 0.091 | 0.039 |
| BearID (face chip) | 0.160 | 0.064 | 0.104 | 0.054 |
