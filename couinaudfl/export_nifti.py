"""Convert institutional format (uint8, WL80/WW225) back to approximate-HU NIfTI, used as input for zero-shot evaluation of public models (e.g., TS 570).

HU ≈ v/255·WW + (WL − WW/2). Values outside the window (below −32.5 or above 192.5) are saturated and cannot be recovered; this is a stated limitation of the evaluation.
Axes: institutional (Z cranial→caudal, Y anterior→posterior, X right→left) → canonical RAS, i.e., the inverse of inst_from_canonical.
"""
from __future__ import annotations
import numpy as np, nibabel as nib

WL, WW = 80.0, 225.0


def u8_to_hu(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0 * WW + (WL - WW / 2)


def inst_to_canonical(a: np.ndarray) -> np.ndarray:
    """Institutional (Z,Y,X) array → canonical RAS (X,Y,Z) array (inverse of inst_from_canonical in prep)."""
    return np.transpose(a, (2, 1, 0))[::-1, ::-1, ::-1]


def save_nifti(arr_inst: np.ndarray, spacing_zyx, path: str, dtype=np.float32):
    sz, sy, sx = spacing_zyx; can = np.ascontiguousarray(inst_to_canonical(arr_inst)).astype(dtype)
    aff = np.diag([sx, sy, sz, 1.0]); nib.save(nib.Nifti1Image(can, aff), path)


def load_nifti_as_inst(path: str) -> np.ndarray:
    c = nib.as_closest_canonical(nib.load(path)); a = np.asarray(c.dataobj)
    return np.transpose(a[::-1, ::-1, ::-1], (2, 1, 0))
