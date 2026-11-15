# Method ladder on the leave-one-year-out splits (stage 6)

Queries: sampled photos of the held-out year whose bear is in training; gallery: sampled train_iid photos. Pooled over 2017-2022. 95% CIs resample bears. Secondary analysis.

## Subset: faces (2094 queries, 77 bears)

| Method | Rank-1 [95% CI] | mAP [95% CI] |
|---|---|---|
| MegaDescriptor (zero-shot) | 0.096 [0.074, 0.118] | 0.048 [0.042, 0.054] |
| BearID (zero-shot) | 0.101 [0.079, 0.125] | 0.059 [0.051, 0.067] |
| MegaDescriptor + head (face subset) | 0.176 [0.137, 0.216] | 0.131 [0.102, 0.159] |
| BearID + head (face subset) | 0.140 [0.111, 0.172] | 0.075 [0.063, 0.087] |
| MegaDescriptor + head (full train split) | 0.237 [0.195, 0.280] | 0.155 [0.130, 0.181] |
| PoseSwin | 0.753 [0.715, 0.790] | 0.782 [0.744, 0.817] |
| PoseSwin, pose-filtered | 0.745 [0.705, 0.781] | 0.773 [0.735, 0.808] |

| Comparison | Δ rank-1 [95% CI] | p | Δ mAP [95% CI] | p |
|---|---|---|---|---|
| BearID (zero-shot) − MegaDescriptor (zero-shot) | +0.005 [-0.014, +0.024] | 0.620 | +0.010 [+0.004, +0.017] | 0.004 |
| MegaDescriptor + head (full train split) − BearID (zero-shot) | +0.136 [+0.100, +0.171] | 0.000 | +0.097 [+0.076, +0.116] | 0.000 |
| PoseSwin − MegaDescriptor + head (full train split) | +0.516 [+0.480, +0.550] | 0.000 | +0.627 [+0.597, +0.655] | 0.000 |
| PoseSwin − BearID (zero-shot) | +0.652 [+0.617, +0.686] | 0.000 | +0.723 [+0.689, +0.754] | 0.000 |
| BearID + head (face subset) − MegaDescriptor + head (face subset) | -0.036 [-0.071, -0.003] | 0.032 | -0.056 [-0.076, -0.036] | 0.000 |
| PoseSwin − MegaDescriptor (zero-shot) | +0.657 [+0.622, +0.690] | 0.000 | +0.734 [+0.698, +0.766] | 0.000 |

Rank-1 by held-out year:

| Year | MegaDescriptor (zero-shot) | BearID (zero-shot) | MegaDescriptor + head (face subset) | BearID + head (face subset) | MegaDescriptor + head (full train split) | PoseSwin | PoseSwin, pose-filtered |
|---|---|---|---|---|---|---|---|
| 2017 | 0.114 | 0.112 | 0.180 | 0.146 | 0.219 | 0.718 | 0.711 |
| 2018 | 0.102 | 0.096 | 0.188 | 0.150 | 0.242 | 0.726 | 0.717 |
| 2019 | 0.102 | 0.105 | 0.191 | 0.151 | 0.256 | 0.762 | 0.756 |
| 2020 | 0.088 | 0.103 | 0.176 | 0.137 | 0.176 | 0.787 | 0.778 |
| 2021 | 0.089 | 0.100 | 0.164 | 0.136 | 0.300 | 0.832 | 0.816 |
| 2022 | 0.077 | 0.085 | 0.157 | 0.117 | 0.210 | 0.653 | 0.653 |

## Subset: all (3720 queries, 77 bears)

| Method | Rank-1 [95% CI] | mAP [95% CI] |
|---|---|---|
| MegaDescriptor (zero-shot) | 0.094 [0.074, 0.115] | 0.037 [0.032, 0.042] |
| BearID (zero-shot) | 0.057 [0.043, 0.071] | 0.033 [0.028, 0.038] |
| MegaDescriptor + head (face subset) | 0.143 [0.114, 0.172] | 0.072 [0.058, 0.086] |
| BearID + head (face subset) | 0.079 [0.061, 0.098] | 0.042 [0.035, 0.050] |
| MegaDescriptor + head (full train split) | 0.225 [0.190, 0.260] | 0.120 [0.101, 0.138] |
| PoseSwin | 0.710 [0.675, 0.744] | 0.740 [0.703, 0.773] |
| PoseSwin, pose-filtered | 0.643 [0.606, 0.678] | 0.666 [0.630, 0.701] |

| Comparison | Δ rank-1 [95% CI] | p | Δ mAP [95% CI] | p |
|---|---|---|---|---|
| BearID (zero-shot) − MegaDescriptor (zero-shot) | -0.037 [-0.054, -0.019] | 0.000 | -0.004 [-0.008, +0.000] | 0.058 |
| MegaDescriptor + head (full train split) − BearID (zero-shot) | +0.168 [+0.137, +0.196] | 0.000 | +0.087 [+0.070, +0.102] | 0.000 |
| PoseSwin − MegaDescriptor + head (full train split) | +0.485 [+0.456, +0.516] | 0.000 | +0.620 [+0.589, +0.646] | 0.000 |
| PoseSwin − BearID (zero-shot) | +0.653 [+0.619, +0.684] | 0.000 | +0.707 [+0.671, +0.737] | 0.000 |
| BearID + head (face subset) − MegaDescriptor + head (face subset) | -0.064 [-0.086, -0.041] | 0.000 | -0.030 [-0.038, -0.020] | 0.000 |
| PoseSwin − MegaDescriptor (zero-shot) | +0.617 [+0.585, +0.645] | 0.000 | +0.703 [+0.668, +0.733] | 0.000 |

Rank-1 by held-out year:

| Year | MegaDescriptor (zero-shot) | BearID (zero-shot) | MegaDescriptor + head (face subset) | BearID + head (face subset) | MegaDescriptor + head (full train split) | PoseSwin | PoseSwin, pose-filtered |
|---|---|---|---|---|---|---|---|
| 2017 | 0.121 | 0.063 | 0.156 | 0.082 | 0.220 | 0.676 | 0.613 |
| 2018 | 0.096 | 0.053 | 0.148 | 0.084 | 0.225 | 0.684 | 0.627 |
| 2019 | 0.093 | 0.054 | 0.140 | 0.078 | 0.239 | 0.718 | 0.627 |
| 2020 | 0.083 | 0.062 | 0.131 | 0.082 | 0.201 | 0.739 | 0.686 |
| 2021 | 0.083 | 0.058 | 0.149 | 0.079 | 0.262 | 0.780 | 0.708 |
| 2022 | 0.077 | 0.048 | 0.125 | 0.066 | 0.182 | 0.639 | 0.570 |

## Rank-1 by head angle (subset: all; methods that cannot embed a photo score it as a miss)

| Head angle | Queries | MegaDescriptor (zero-shot) | BearID (zero-shot) | MegaDescriptor + head (face subset) | BearID + head (face subset) | MegaDescriptor + head (full train split) | PoseSwin | PoseSwin, pose-filtered |
|---|---|---|---|---|---|---|---|---|
| eye_undetected | 251 | 0.080 [0.04, 0.12] | 0.000 [0.00, 0.00] | 0.068 [0.03, 0.10] | 0.000 [0.00, 0.00] | 0.139 [0.08, 0.20] | 0.506 [0.43, 0.58] | 0.040 [0.02, 0.06] |
| frontal | 774 | 0.103 [0.07, 0.13] | 0.118 [0.08, 0.15] | 0.187 [0.14, 0.23] | 0.176 [0.13, 0.22] | 0.249 [0.20, 0.29] | 0.722 [0.67, 0.77] | 0.713 [0.66, 0.76] |
| profile | 1546 | 0.087 [0.06, 0.11] | 0.012 [0.01, 0.02] | 0.114 [0.08, 0.15] | 0.012 [0.01, 0.02] | 0.207 [0.16, 0.25] | 0.699 [0.65, 0.74] | 0.626 [0.58, 0.67] |
| three_quarter | 1149 | 0.100 [0.08, 0.12] | 0.089 [0.07, 0.11] | 0.169 [0.13, 0.21] | 0.121 [0.10, 0.15] | 0.252 [0.21, 0.29] | 0.763 [0.72, 0.80] | 0.749 [0.71, 0.79] |

Trained heads (validation accuracy used to pick the epoch):

- 2017: MD head val acc 0.873 on 41814 training photos; face-subset heads (holdout acc) MD 0.473, BearID 0.330 on 1129 photos
- 2018: MD head val acc 0.861 on 41123 training photos; face-subset heads (holdout acc) MD 0.528, BearID 0.360 on 1256 photos
- 2019: MD head val acc 0.864 on 38661 training photos; face-subset heads (holdout acc) MD 0.488, BearID 0.421 on 1215 photos
- 2020: MD head val acc 0.864 on 37639 training photos; face-subset heads (holdout acc) MD 0.535, BearID 0.378 on 1278 photos
- 2021: MD head val acc 0.875 on 36702 training photos; face-subset heads (holdout acc) MD 0.509, BearID 0.393 on 1123 photos
- 2022: MD head val acc 0.857 on 35986 training photos; face-subset heads (holdout acc) MD 0.473, BearID 0.333 on 1299 photos
