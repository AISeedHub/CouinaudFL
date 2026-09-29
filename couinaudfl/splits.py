"""Public-data splits, designed to avoid train/test contamination.

Rules
  * Lee test-50 (MSD Task08, Tian labels) is never used for training → external test A.
  * The 41 MedSeg-unique cases (Task08) are excluded from training under all label sources (Tian, NIH, MedSeg) → external test B (9 segments).
  * Multiple labelings of the same image (Tian/NIH) are kept on the same side (train/val/fold), per same_image_groups.json.
  * MR (06) and CRLM (07) are not used for training (cross-modality / external-cohort tests).
Modes
  * "pretrain": group-level train/val split (val_frac) after the exclusions above
  * "tian5fold": 5-fold CV over all 193 Tian cases (same protocol as TS 570, for comparison with its 0.897). Excluding the 41 MedSeg cases is optional.
"""
from __future__ import annotations
import os, json, random

LEE_TEST50 = {1, 4, 9, 10, 11, 13, 15, 21, 27, 44, 50, 52, 53, 62, 66, 69, 71, 75, 82, 89, 91, 92, 101, 116, 117, 124, 136, 140,
              147, 171, 179, 183, 213, 215, 229, 245, 265, 269, 275, 287, 305, 307, 359, 375, 377, 399, 425, 441, 445, 455}
MEDSEG_TASK08 = {1, 2, 4, 5, 7, 8, 10, 11, 13, 16, 18, 19, 20, 22, 26, 27, 29, 30, 31, 32, 39, 40, 42, 44, 49, 50, 51, 52, 53, 57,
                 58, 59, 61, 62, 65, 67, 68, 72, 75, 77, 78}   # MedSeg 50 → 41 unique cases (matched by voxel hash)
TRAIN_SETS = ["01_msd08_tian", "02_msd08_nih", "03_lits_zhang", "04_ircadb_zhang"]
CRLM_EXCLUDE = {"CRLM-CT-1075"}   # voxel-identical to hepaticvessel_337 (training pool; same MSKCC image) → excluded from CRLM evaluation


def task08_id(case: str) -> int | None:
    return int(case.split("_")[-1]) if case.startswith("hepaticvessel_") else None


def load_groups(root: str) -> dict[str, str]:
    """Case path ('ds/case') → group id; ungrouped cases map to themselves."""
    g = json.load(open(os.path.join(root, "same_image_groups.json")))
    m = {}
    for gid, members in g.items():
        for x in members: m[x] = gid
    return m


def public_pool(root: str, sets=TRAIN_SETS, exclude_lee=True, exclude_medseg=True) -> list[str]:
    out = []
    for ds in sets:
        for c in sorted(os.listdir(os.path.join(root, ds))):
            if not os.path.isdir(os.path.join(root, ds, c)): continue
            t8 = task08_id(c)
            if t8 is not None and ((exclude_lee and t8 in LEE_TEST50) or (exclude_medseg and t8 in MEDSEG_TASK08)): continue
            out.append(f"{ds}/{c}")
    return out


def group_split(root: str, cases: list[str], val_frac: float = 0.1, seed: int = 42):
    gm = load_groups(root); groups = {}
    for c in cases: groups.setdefault(gm.get(c, c), []).append(c)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    nv = max(1, int(round(len(keys) * val_frac)))
    val = sorted(x for k in keys[:nv] for x in groups[k]); train = sorted(x for k in keys[nv:] for x in groups[k])
    return train, val


def group_kfold(root: str, cases: list[str], n_folds: int = 5, seed: int = 42):
    gm = load_groups(root); groups = {}
    for c in cases: groups.setdefault(gm.get(c, c), []).append(c)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    return [sorted(x for k in keys[f::n_folds] for x in groups[k]) for f in range(n_folds)]


def make_pretrain_split(root: str, val_frac=0.1, seed=42) -> dict:
    pool = public_pool(root)
    train, val = group_split(root, pool, val_frac, seed)
    lee = [f"01_msd08_tian/hepaticvessel_{i:03d}" for i in sorted(LEE_TEST50)]
    medseg = [f"05_msd08_medseg/{c}" for c in sorted(os.listdir(os.path.join(root, "05_msd08_medseg"))) if os.path.isdir(os.path.join(root, "05_msd08_medseg", c))]
    return {"mode": "pretrain", "train": train, "val": val, "test_lee50_tian": lee,
            "test_lee50_nih": [x.replace("01_msd08_tian", "02_msd08_nih") for x in lee], "test_medseg": medseg,
            "test_mr": [f"06_ts_mr/{c}" for c in sorted(os.listdir(os.path.join(root, "06_ts_mr"))) if os.path.isdir(os.path.join(root, "06_ts_mr", c))],
            "test_crlm": [f"07_crlm/{c}" for c in sorted(os.listdir(os.path.join(root, "07_crlm"))) if os.path.isdir(os.path.join(root, "07_crlm", c)) and c not in CRLM_EXCLUDE],
            "excluded": {"lee50": len(LEE_TEST50), "medseg_task08": len(MEDSEG_TASK08)}}


def make_tian5fold_split(root: str, exclude_medseg=True, seed=42) -> dict:
    tian = public_pool(root, sets=["01_msd08_tian"], exclude_lee=False, exclude_medseg=exclude_medseg)
    folds = group_kfold(root, tian, 5, seed)
    return {"mode": "tian5fold", "folds": folds, "n": len(tian), "exclude_medseg": exclude_medseg}


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "path/to/public_data"
    s = make_pretrain_split(root); f = make_tian5fold_split(root)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items()})
    print("tian5fold:", f["n"], [len(x) for x in f["folds"]])
    os.makedirs("outputs/splits", exist_ok=True)
    json.dump(s, open("outputs/splits/pretrain_split.json", "w"), indent=1); json.dump(f, open("outputs/splits/tian5fold_split.json", "w"), indent=1)
