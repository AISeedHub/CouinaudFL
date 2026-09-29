"""Lesion-to-segment localization: builds a per-lesion overlap table (anonymous numbers only) locally at each site.

Each row is one lesion connected component. For both GT segments (ch1–9) and the predicted label map we record
  frac(s) = |lesion ∩ segment s| / |lesion|
A lesion is assigned to segment s by thresholding this fraction (the threshold is swept in server-side analysis);
gt_frac is the GT-based fraction and pred_frac the model-based one.
"""
from __future__ import annotations
import os
import numpy as np
from scipy import ndimage
from .data import SEG_NAMES, NUM_CLASSES

MIN_LESION_ML = 0.05   # smaller components are treated as noise


def load_lesions(case_dir: str):
    """mask.npy ch10–17 → (lesion union bool (D,H,W), channel index map), or None if no lesion channels."""
    p = os.path.join(case_dir, "mask.npy")
    if not os.path.exists(p): return None
    m = np.load(p, mmap_mode="r")
    if m.shape[0] < 11: return None
    les = np.asarray(m[10:min(18, m.shape[0])])
    if les.sum() == 0: return None
    ch = np.zeros(les.shape[1:], np.int16)
    for k in range(les.shape[0]):
        ch[les[k] > 0] = 10 + k
    return les.any(0), ch


def lesion_overlap_rows(case: str, case_dir: str, gt_label: np.ndarray, pred_label: np.ndarray, spacing) -> list[dict]:
    lz = load_lesions(case_dir)
    if lz is None: return []
    les, chmap = lz
    D = min(les.shape[0], gt_label.shape[0], pred_label.shape[0])
    les, chmap = les[:D], chmap[:D]; gt, pr = gt_label[:D], pred_label[:D]
    vox_ml = float(np.prod(spacing)) / 1000.0
    comp, nc = ndimage.label(les)
    rows = []
    for li in range(1, nc + 1):
        sel = comp == li; v = int(sel.sum())
        if v * vox_ml < MIN_LESION_ML: continue
        r = {"case": case, "lesion_id": li, "lesion_ml": round(v * vox_ml, 3),
             "lesion_channel": int(np.bincount(chmap[sel]).argmax())}
        g = gt[sel]; p = pr[sel]
        # single-segment assignment at the lesion center (lesion voxel closest to the centroid), matching the clinical notion of location
        idx = np.argwhere(sel); com = idx.mean(0); ci = idx[np.argmin(((idx - com) ** 2).sum(1))]
        r["com_gt_seg"] = int(gt[tuple(ci)]); r["com_pred_seg"] = int(pr[tuple(ci)])
        for c in range(1, NUM_CLASSES):
            n = SEG_NAMES[c - 1]
            r[f"gt_frac_{n}"] = round(float((g == c).sum() / v), 4)
            r[f"pred_frac_{n}"] = round(float((p == c).sum() / v), 4)
        rows.append(r)
    return rows
