# Same model, different crop (stage 9)

MegaDescriptor on the body crop, on BearID's face box (15% margin) and on BearID's aligned chip, next to BearID, on the stage-4 primary frames and pairs. Designed after stage 8; secondary.

## face_w>=24px (885 images, 386 sequences, 13 bears)

| Negatives | MD, body crop | MD, face box | MD, aligned face chip | BearID (face chip) | Fused (MD body + BearID) |
|---|---|---|---|---|---|
| all | 0.768 [0.747, 0.790] | 0.627 [0.606, 0.649] | 0.641 [0.621, 0.661] | 0.660 [0.639, 0.680] | 0.781 [0.762, 0.800] |
| within-zoo | 0.622 [0.592, 0.655] | 0.587 [0.559, 0.610] | 0.580 [0.554, 0.604] | 0.618 [0.589, 0.644] | 0.664 [0.634, 0.692] |

| Negatives | Comparison | Unit | Δ [95% CI] | p |
|---|---|---|---|---|
| all | MD, face box − MD, body crop | sequence | -0.140 [-0.166, -0.115] | <0.001 |
| all | MD, aligned face chip − MD, body crop | sequence | -0.126 [-0.151, -0.103] | <0.001 |
| all | MD, aligned face chip − MD, face box | sequence | +0.014 [+0.001, +0.026] | 0.022 |
| all | BearID (face chip) − MD, aligned face chip | sequence | +0.019 [-0.005, +0.040] | 0.130 |
| all | BearID (face chip) − MD, face box | sequence | +0.032 [+0.008, +0.054] | 0.010 |
| all | BearID (face chip) − MD, body crop | sequence | -0.108 [-0.139, -0.081] | <0.001 |
| all | MD, face box − MD, body crop | bear | -0.140 [-0.199, -0.082] | <0.001 |
| all | MD, aligned face chip − MD, body crop | bear | -0.126 [-0.177, -0.072] | <0.001 |
| all | MD, aligned face chip − MD, face box | bear | +0.014 [-0.005, +0.029] | 0.148 |
| all | BearID (face chip) − MD, aligned face chip | bear | +0.019 [-0.022, +0.039] | 0.326 |
| all | BearID (face chip) − MD, face box | bear | +0.032 [-0.009, +0.057] | 0.130 |
| all | BearID (face chip) − MD, body crop | bear | -0.108 [-0.169, -0.060] | <0.001 |
| within-zoo | MD, face box − MD, body crop | sequence | -0.036 [-0.072, -0.002] | 0.048 |
| within-zoo | MD, aligned face chip − MD, body crop | sequence | -0.042 [-0.078, -0.010] | 0.004 |
| within-zoo | MD, aligned face chip − MD, face box | sequence | -0.007 [-0.022, +0.010] | 0.412 |
| within-zoo | BearID (face chip) − MD, aligned face chip | sequence | +0.038 [+0.008, +0.065] | 0.012 |
| within-zoo | BearID (face chip) − MD, face box | sequence | +0.032 [-0.000, +0.063] | 0.052 |
| within-zoo | BearID (face chip) − MD, body crop | sequence | -0.004 [-0.047, +0.036] | 0.794 |
| within-zoo | MD, face box − MD, body crop | bear | -0.036 [-0.178, +0.044] | 0.428 |
| within-zoo | MD, aligned face chip − MD, body crop | bear | -0.042 [-0.131, +0.035] | 0.319 |
| within-zoo | MD, aligned face chip − MD, face box | bear | -0.007 [-0.027, +0.070] | 0.602 |
| within-zoo | BearID (face chip) − MD, aligned face chip | bear | +0.038 [-0.150, +0.087] | 0.685 |
| within-zoo | BearID (face chip) − MD, face box | bear | +0.032 [-0.108, +0.083] | 0.667 |
| within-zoo | BearID (face chip) − MD, body crop | bear | -0.004 [-0.232, +0.108] | 0.888 |

| Model | Rank-1 | mAP |
|---|---|---|
| MD, body crop | 0.812 | 0.442 |
| MD, face box | 0.669 | 0.285 |
| MD, aligned face chip | 0.729 | 0.298 |
| BearID (face chip) | 0.632 | 0.298 |
| Fused (MD body + BearID) | 0.863 | 0.468 |
