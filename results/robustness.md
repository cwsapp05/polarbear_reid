# Robustness checks (face_w>=24px)

Same 885 frames as the stage-4 headline subset. All of this was planned after the stage-4 results were seen, so treat it as secondary.

## A. Who is in the face-visible subset

| Zoo | Bear | Images | Seq. | Image share | Positive-pair share |
|---|---|---|---|---|---|
| Tiergarten Nürnberg | Vera | 183 | 76 | 20.7% | 30.8% |
| Tierpark Neumünster | Larissa | 171 | 66 | 19.3% | 26.9% |
| Tierpark Berlin | Tonja | 147 | 55 | 16.6% | 19.8% |
| Tierpark Neumünster | Vitus | 101 | 42 | 11.4% | 9.3% |
| Tierpark Berlin | Hertha | 88 | 48 | 9.9% | 7.0% |
| Tiergarten Nürnberg | Nanuq | 61 | 26 | 6.9% | 3.3% |
| Tiergarten Schönbrunn | Nora | 40 | 18 | 4.5% | 1.4% |
| Zoo Hannover | Nana | 28 | 16 | 3.2% | 0.7% |
| Zoo Hannover | Milana | 23 | 11 | 2.6% | 0.4% |
| Tiergarten Schönbrunn | Finja | 15 | 10 | 1.7% | 0.2% |
| Zoo Karlsruhe | Charlotte | 11 | 7 | 1.2% | 0.1% |
| Zoo Karlsruhe | Lloyd | 10 | 7 | 1.1% | 0.1% |
| Zoo Hannover | Sprinter | 7 | 4 | 0.8% | 0.0% |

The top three bears account for 77% of positive pairs.

## B. Sequence-level vs bear-level bootstrap

| Negatives | Bootstrap unit | MegaDescriptor | BearID | Fused | Δ BearID − MD | p | Δ Fused − MD | p |
|---|---|---|---|---|---|---|---|---|
| all | sequence | 0.768 [0.747, 0.790] | 0.660 [0.639, 0.680] | 0.781 [0.762, 0.800] | -0.108 [-0.139, -0.081] | <0.001 | +0.014 [-0.001, +0.028] | 0.064 |
| all | bear | 0.768 [0.712, 0.821] | 0.660 [0.617, 0.693] | 0.781 [0.719, 0.825] | -0.108 [-0.169, -0.060] | <0.001 | +0.014 [-0.015, +0.034] | 0.308 |
| within-zoo | sequence | 0.622 [0.592, 0.655] | 0.618 [0.589, 0.644] | 0.664 [0.634, 0.692] | -0.004 [-0.047, +0.036] | 0.794 | +0.041 [+0.020, +0.061] | <0.001 |
| within-zoo | bear | 0.622 [0.558, 0.822] | 0.618 [0.520, 0.686] | 0.664 [0.615, 0.780] | -0.004 [-0.232, +0.108] | 0.888 | +0.041 [-0.057, +0.099] | 0.614 |

With only 13 bears, the bear-level intervals are the honest ones for generalising to new bears; the sequence-level intervals only cover new footage of these 13 bears.

## C. Bear-balanced AUROC and leave-one-bear-out

Balanced = every bear's positive pairs and every bear pair's negatives carry equal total weight.

| Negatives | MegaDescriptor | BearID | Fused | Δ BearID − MD | Δ Fused − MD |
|---|---|---|---|---|---|
| all | 0.700 | 0.625 | 0.718 | -0.075 | +0.018 |
| within-zoo | 0.635 | 0.565 | 0.635 | -0.070 | +0.000 |

Leave-one-bear-out (point estimates; range over the 13 drops):

- all, Δ bearid − MD: -0.121 (without Hertha) to -0.093 (without Larissa)
- all, Δ fused − MD: +0.008 (without Vitus) to +0.017 (without Milana)
- within-zoo, Δ bearid − MD: -0.047 (without Hertha) to +0.046 (without Vera)
- within-zoo, Δ fused − MD: +0.022 (without Tonja) to +0.066 (without Vera)

## D. Fusion weight tuned leave-one-zoo-out

Weight on MegaDescriptor chosen from {0, 0.05, …, 1} on the other five zoos, applied to the held-out zoo. Bootstrap is over sequences and conditional on the chosen weights.

**all negatives.** Weights on MD per held-out zoo: Tiergarten Nürnberg: 0.65, Tiergarten Schönbrunn: 0.65, Tierpark Berlin: 0.70, Tierpark Neumünster: 0.65, Zoo Hannover: 0.65, Zoo Karlsruhe: 0.65.

| Score | AUROC [95% CI] | Δ tuned − this [95% CI] | p |
|---|---|---|---|
| megadescriptor | 0.768 [0.747, 0.790] | +0.024 [+0.015, +0.033] | <0.001 |
| bearid | 0.660 [0.639, 0.680] | +0.132 [+0.112, +0.155] | <0.001 |
| fused (equal weight) | 0.781 [0.762, 0.800] | +0.010 [+0.005, +0.016] | <0.001 |
| tuned (out-of-fold) | 0.792 [0.773, 0.811] | – | – |

**within-zoo negatives.** Weights on MD per held-out zoo: Tiergarten Nürnberg: 0.40, Tiergarten Schönbrunn: 0.50, Tierpark Berlin: 0.55, Tierpark Neumünster: 0.60, Zoo Hannover: 0.50, Zoo Karlsruhe: 0.50.

| Score | AUROC [95% CI] | Δ tuned − this [95% CI] | p |
|---|---|---|---|
| megadescriptor | 0.622 [0.592, 0.655] | +0.030 [+0.010, +0.049] | 0.004 |
| bearid | 0.618 [0.589, 0.644] | +0.034 [+0.010, +0.060] | <0.001 |
| fused (equal weight) | 0.664 [0.634, 0.692] | -0.011 [-0.016, -0.007] | <0.001 |
| tuned (out-of-fold) | 0.652 [0.623, 0.680] | – | – |
