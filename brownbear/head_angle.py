"""Head angle of every sampled crop from the authors' own head-keypoint model (HRNet-w48, 13 keypoints).

Run with tools/poseswin-venv. offset = |nose_x - eye_midpoint_x| / eye distance.
  frontal         offset < 0.25
  three_quarter   0.25 <= offset < 1.0
  profile         offset >= 1.0
  eye_undetected  an eye keypoint has confidence < 0.5 (mostly blurred or back-of-head crops)
A first rule (profile = an eye below 0.5 confidence) was revised after a visual check of sample crops
showed the model keeps high confidence on the hidden eye in clear side views; the revision used images
only, before any results by angle were computed.
Output: work/head_angle.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import poseswin_embed as P  # noqa: E402  (sets up the repo path and config)
from project.models.build_swint import get_hrnet_model  # noqa: E402

KP = ['left_eye_L', 'left_eye_R', 'left_eye', 'right_eye_L', 'right_eye_R', 'right_eye', 'nose',
      'lip_right', 'lip_left', 'lip_upper', 'lip_lower', 'left_ear', 'right_ear']
LE, RE, NOSE = KP.index("left_eye"), KP.index("right_eye"), KP.index("nose")
VIS, FRONTAL, PROFILE = 0.5, 0.25, 1.0


def classify(xy, conf):
    le, re = conf[LE] >= VIS, conf[RE] >= VIS
    if not (le and re):
        return "eye_undetected", np.nan
    d = abs(xy[LE, 0] - xy[RE, 0])
    off = abs(xy[NOSE, 0] - (xy[LE, 0] + xy[RE, 0]) / 2) / max(d, 1e-6)
    return ("frontal" if off < FRONTAL else "three_quarter" if off < PROFILE else "profile"), off


def main():
    c = P.cfg.clone()
    c.merge_from_list(["MODEL.POSE_WEIGHT", str(P.DATA / "checkpoints/preprocessing_ckpts/pose/hrnet_w48_balanced_n13_refined.pth"),
                       "MODEL.DEVICE_ID", "('0',)"])
    pose = get_hrnet_model(c, max_batch_size=32)
    smp = pd.read_csv(HERE / "work" / "sample.csv")
    rows = []
    for k in tqdm(range(0, len(smp), 32), desc="keypoints"):
        part = smp.iloc[k:k + 32]
        ims = np.stack([np.asarray(Image.open(p).convert("RGB").resize((256, 256))) for p in part["path"]])
        with torch.no_grad():
            _, _, preds, maxvals = pose.predict(ims)
        for iid, xy, cf in zip(part["image_id"], preds, maxvals):
            cf = np.asarray(cf).reshape(-1)
            b, off = classify(np.asarray(xy).reshape(-1, 2), cf)
            rows.append({"image_id": iid, "head_angle": b, "nose_offset": off,
                         "conf_left_eye": cf[LE], "conf_right_eye": cf[RE], "conf_mean": cf.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "work" / "head_angle.csv", index=False)
    print(out["head_angle"].value_counts().to_dict())


if __name__ == "__main__":
    main()
