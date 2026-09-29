"""Data layer for the institutional format (image.npy uint8 (D,512,512) + mask.npy (C,D,512,512)).

Label convention (10-class softmax): 0 background, 1 S1, 2 S2, 3 S3, 4 S4a, 5 S4b, 6 S5, 7 S6, 8 S7, 9 S8
  - Institutional mask.npy: 18 channels (ch0 background, ch1–9 segments, ch10–17 lesions) → argmax over ch0..9; lesion channels ignored
  - Public mask.npy: 10 channels (ch0 background, ch1–9). For 8-segment data ch5 (S4b) is empty and ch4 holds all of S4.
    Such cases have segments==8 in meta.json; the loss sums the 4a/4b probabilities and supervises them as a single S4 (partial labels).
Training patches: full z (80) × 256 × 256, preferentially sampled around the liver. No left-right flip augmentation (Couinaud anatomy is chiral).
"""
from __future__ import annotations
import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 10
SEG_NAMES = ["S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7", "S8"]
PATCH = (96, 256, 256)   # TS architecture downsamples 5 times (÷32) → z 80 is zero-padded to 96 (volumes stay at 80 slices)
MEAN, STD = 0.5, 0.25          # fixed normalization on uint8/255 (same WL80/WW225 window for institutional and public data)


def normalize(img_u8: np.ndarray) -> np.ndarray:
    return ((img_u8.astype(np.float32) / 255.0) - MEAN) / STD


DEFAULT_SPACING = [4.0, 0.73046875, 0.73046875]   # z fixed at 4 mm; in-plane from a reference site (per-site spacing.csv recommended)
_SPACING_CACHE: dict = {}


SPACING_CSV = os.environ.get("COUINAUD_SPACING_CSV", "")   # optional: spacing file outside the data root


def set_spacing_file(path: str):
    """For --spacing-file: all subsequent load_case calls use this file (csv/xlsx) with top priority."""
    global SPACING_CSV
    SPACING_CSV = path or SPACING_CSV
    _SPACING_CACHE.clear()


def _norm_col(c):  # normalize column names (e.g., Resolution (512, 1024) → resolution)
    c = str(c).strip().lower()
    for key, names in {"id": ["id", "case", "patient_id"], "resolution": ["resolution"], "pixel": ["pixel spacing", "pixel_spacing", "pixelspacing"],
                       "thick": ["slice thickeness", "slice thickness", "thickness"], "incr": ["incremental", "increment"]}.items():
        if any(c.startswith(n) for n in names): return key
    return c


def _rows_from_xlsx(path):
    import pandas as pd
    df = pd.read_excel(path, dtype=str)   # preserve leading zeros and letters in IDs
    df.columns = [_norm_col(c) for c in df.columns]
    return df.to_dict("records")


def _spacing_table(root: str) -> dict:
    """Per-patient spacing table → {case: [z, s, s]} (in-plane s at 512 matrix, per-patient z).
    File search order: $COUINAUD_SPACING_CSV → <data>/spacing.(xlsx|csv) → <data>/volumes_per_patient.csv → a single <data>/*.xlsx.
    xlsx columns (site spreadsheet format): Matching, ID, Resolution(512|1024), Pixel spacing, Slice thickeness, Incremental[, Date]
      · in-plane s = Pixel spacing × Resolution/512 (×2 correction for 1024 matrices)
      · z = Incremental, else Slice thickness, else 4.0 (for sites leaving the field blank)
      · IDs are matched as strings (leading zeros and letters preserved), exact match with the folder name"""
    if root in _SPACING_CACHE: return _SPACING_CACHE[root]
    cands = [SPACING_CSV, os.path.join(root, "spacing.xlsx"), os.path.join(root, "spacing.csv"), os.path.join(root, "volumes_per_patient.csv")]
    import glob as _g
    xs = [p for p in _g.glob(os.path.join(root, "*.xlsx")) if not os.path.basename(p).startswith("~")]
    if len(xs) == 1: cands.append(xs[0])
    tbl = {}
    for p in cands:
        if not p or not os.path.exists(p): continue
        rows = []
        if p.lower().endswith(".xlsx"):
            try: rows = _rows_from_xlsx(p)
            except Exception: continue
        else:
            import csv
            rows = [{ _norm_col(k): v for k, v in r.items()} for r in csv.DictReader(open(p, encoding="utf-8-sig"))]
        for r in rows:
            k = str(r.get("id") or "").strip()
            if not k or k.lower() == "nan" or k in tbl: continue
            try:
                def num(key):
                    v = r.get(key)
                    v = "" if v is None else str(v).strip()
                    if not v or v.lower() == "nan": return None
                    try: return float(v)
                    except ValueError: return None   # non-numeric values such as '?' or '100W116 (...)' → missing
                if r.get("spacing_y"):   # simple csv format
                    tbl[k] = [num("spacing_z") or 4.0, float(r["spacing_y"]), float(r["spacing_x"])]; continue
                px = num("pixel") if "pixel" in r else num("pixel_spacing")
                res = num("resolution") or num("orig_matrix") or 512.0
                sp_in = (px * res / 512.0) if px is not None else None
                if sp_in is not None and not (0.3 <= sp_in <= 1.5): sp_in = None   # implausible values such as '2' or strings → missing
                z = num("incr") or num("thick") or 4.0
                if not (1.0 <= z <= 10.0): z = 4.0
                if sp_in is None and z == 4.0 and (px is not None):   # invalid in-plane and default z → treated as invalid (z already applied)
                    pass
                tbl[k] = [z, sp_in if sp_in is not None else -1.0, sp_in if sp_in is not None else -1.0]
            except (KeyError, ValueError, TypeError): continue
        if tbl: break
    # Note: IDs use exact string matching only. Some patients share the numeric part and differ only by a suffix (e.g., ...a1),
    # so approximate matching (e.g., stripping leading zeros) risks misassignment. Unmatched cases show spacing_source=default.
    _SPACING_CACHE[root] = tbl; return tbl


LABEL_CACHE = os.environ.get("COUINAUD_LABEL_CACHE", "")   # cache location if the data folder should not be written to


def _label_path(case_dir: str) -> str:
    if LABEL_CACHE:
        p = os.path.join(LABEL_CACHE, os.path.basename(os.path.dirname(case_dir.rstrip("/"))), os.path.basename(case_dir.rstrip("/")))
        os.makedirs(p, exist_ok=True); return os.path.join(p, "label.npy")
    return os.path.join(case_dir, "label.npy")


def load_case(case_dir: str):
    """→ (image uint8 (D,H,W), label uint8 (D,H,W), meta dict).
    mask.npy (10–18 channels, 210–380 MB per case) is argmaxed once and cached as label.npy (21 MB) to remove the I/O bottleneck."""
    img = np.load(os.path.join(case_dir, "image.npy"), mmap_mode="r")
    mp = os.path.join(case_dir, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    if "spacing" not in meta:   # spacing for volumes (mL): (1) patient meta.json, (2) spacing.csv in data root, (3) default (0.73 mm)
        sp = _spacing_table(os.path.dirname(case_dir.rstrip("/"))).get(os.path.basename(case_dir.rstrip("/")))
        if sp is not None and sp[1] > 0:
            meta["spacing"] = [float(sp[0]), float(sp[1]), float(sp[2])]; meta["spacing_source"] = "spacing_table"
        elif sp is not None:   # listed but with invalid in-plane value (e.g., '2', text) → z from table, default in-plane
            meta["spacing"] = [float(sp[0]), DEFAULT_SPACING[1], DEFAULT_SPACING[2]]; meta["spacing_source"] = "table_invalid_inplane"
        elif "orig_spacing" in meta: meta["spacing"] = [4.0, float(meta["orig_spacing"][1]), float(meta["orig_spacing"][2])]; meta["spacing_source"] = "meta.orig_spacing"   # public preprocessed data are resampled to z = 4 mm
        else: meta["spacing"] = DEFAULT_SPACING; meta["spacing_source"] = "default"
    else: meta.setdefault("spacing_source", "meta.json")
    lp = _label_path(case_dir)
    if os.path.exists(lp):
        lab = np.load(lp)
    else:
        msk = np.load(os.path.join(case_dir, "mask.npy"), mmap_mode="r")
        lab = np.argmax(np.asarray(msk[:NUM_CLASSES]), axis=0).astype(np.uint8)
        try: np.save(lp, lab)
        except OSError: pass
    D = min(img.shape[0], lab.shape[0])
    return np.asarray(img[:D]), lab[:D], meta


def is_8seg(meta: dict) -> bool:
    return int(meta.get("segments", 9)) == 8


class CouinaudDataset(Dataset):
    """mode='train': random patches with augmentation; 'eval': full volumes (for sliding-window inference)."""

    def __init__(self, case_dirs: list[str], mode: str = "train", patch=PATCH, fg_prob: float = 0.8,
                 augment: bool = True, cache: bool = False):
        self.cases = list(case_dirs); self.mode = mode; self.patch = tuple(patch)
        self.fg_prob = fg_prob; self.augment = augment and mode == "train"
        self._cache = {} if cache else None
        self._aug = self._build_aug() if self.augment else None

    def __len__(self): return len(self.cases)

    def _get(self, i):
        if self._cache is not None and i in self._cache: return self._cache[i]
        item = load_case(self.cases[i])
        if self._cache is not None: self._cache[i] = item
        return item

    @staticmethod
    def _build_aug():
        from monai import transforms as T
        return T.Compose([
            T.RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0, 0, np.deg2rad(10)),
                          scale_range=(0, 0.1, 0.1), mode=("bilinear", "nearest"), padding_mode="zeros"),
            T.RandScaleIntensityd(keys="image", factors=0.15, prob=0.5),
            T.RandShiftIntensityd(keys="image", offsets=0.15, prob=0.5),
            T.RandAdjustContrastd(keys="image", gamma=(0.8, 1.25), prob=0.3),
            T.RandGaussianNoised(keys="image", std=0.05, prob=0.2),
            T.RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.0, 0.5), prob=0.15),
        ])

    def _crop(self, img, lab):
        D, H, W = lab.shape; pd, ph, pw = self.patch
        # z: zero-pad if the patch exceeds the volume, otherwise random start
        if random.random() < self.fg_prob and lab.any():
            zs, ys, xs = np.nonzero(lab[::4, ::8, ::8]); k = random.randrange(len(zs))
            cz, cy, cx = zs[k] * 4, ys[k] * 8, xs[k] * 8
            z0 = int(np.clip(cz - pd // 2 + random.randint(-pd // 4, pd // 4), 0, max(D - pd, 0)))
            y0 = int(np.clip(cy - ph // 2 + random.randint(-ph // 4, ph // 4), 0, max(H - ph, 0)))
            x0 = int(np.clip(cx - pw // 2 + random.randint(-pw // 4, pw // 4), 0, max(W - pw, 0)))
        else:
            z0 = random.randint(0, max(D - pd, 0)); y0 = random.randint(0, max(H - ph, 0)); x0 = random.randint(0, max(W - pw, 0))
        ci = img[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]; cl = lab[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        if ci.shape != self.patch:
            pi = np.zeros(self.patch, ci.dtype); pl = np.zeros(self.patch, cl.dtype)
            pi[:ci.shape[0], :ci.shape[1], :ci.shape[2]] = ci; pl[:cl.shape[0], :cl.shape[1], :cl.shape[2]] = cl
            ci, cl = pi, pl
        return ci, cl

    def __getitem__(self, i):
        img, lab, meta = self._get(i)
        if self.mode == "train":
            img, lab = self._crop(img, lab)
        x = normalize(img)[None]; y = lab.astype(np.int64)
        if self._aug is not None:
            d = self._aug({"image": torch.from_numpy(x), "label": torch.from_numpy(y)[None]})
            x = d["image"].float(); y = d["label"][0].long()
        else:
            x = torch.from_numpy(np.ascontiguousarray(x)); y = torch.from_numpy(y)
        return {"image": x, "label": y, "seg8": torch.tensor(is_8seg(meta)), "case": os.path.basename(self.cases[i]),
                "spacing": torch.tensor(meta.get("spacing", meta.get("orig_spacing", [4.0, 0.7, 0.7]))[:3] if meta else [4.0, 0.7, 0.7], dtype=torch.float32)}


def list_cases(root: str) -> list[str]:
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and os.path.exists(os.path.join(root, d, "image.npy")))


def kfold_split(ids: list[str], fold: int, n_folds: int = 5, seed: int = 42, val_frac: float = 0.1):
    """Sort → seeded shuffle → round-robin assignment. test = fold k; val = val_frac of the remainder."""
    ids = sorted(ids); rng = random.Random(seed); rng.shuffle(ids)
    folds = [ids[k::n_folds] for k in range(n_folds)]
    test = folds[fold]; rest = [p for k in range(n_folds) if k != fold for p in folds[k]]
    rng2 = random.Random(seed + 1000 + fold); rng2.shuffle(rest)
    nv = max(1, int(round(len(rest) * val_frac)))
    return sorted(rest[nv:]), sorted(rest[:nv]), sorted(test)
