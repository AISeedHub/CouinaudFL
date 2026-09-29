#!/usr/bin/env python
r"""Patient catalog: records metadata and ground-truth volumes for all patients at a site in a single pass (no later revisit needed).

  python scripts/patient_catalog.py --data <data_root> --exp exp4c --site A
→ outputs/<exp>/client_<site>/patient_catalog.csv  (one row per patient; contains real IDs, replaced with Case N on export)
Columns: case, n_slices, spacing_z/y/x(mm), voxel_ml, liver_ml, seg{S1..S8}_ml, seg_present_count, lesion_ch10..17_ml, lesion_total_ml,
    liver_z0, liver_z1 (slice range containing the liver), liver_median_u8, image_mean_u8, fold_test (fold in which the case is a test case)
"""
import os, sys, json, argparse, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, yaml
from couinaudfl.data import list_cases, load_case, kfold_split, SEG_NAMES


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--exp", required=True); ap.add_argument("--site", default=socket.gethostname())
    ap.add_argument("--config", default=None); ap.add_argument("--out", default="outputs"); ap.add_argument("--run", required=True); ap.add_argument("--spacing-file", default=None, help="Path to the per-patient voxel spacing table (csv/xlsx)"); ap.add_argument("--lesion-labels", default=None, help="Path to the lesion channel mapping json (highest priority)"); a = ap.parse_args()
    C = yaml.safe_load(open(a.config or f"configs/{a.exp}.yaml", encoding="utf-8")); root = os.path.join(a.out, a.exp, a.run, f"client_{a.site}"); os.makedirs(root, exist_ok=True)
    from couinaudfl.data import set_spacing_file
    if a.spacing_file: set_spacing_file(a.spacing_file)
    cases = list_cases(a.data); fold_of = {}
    for k in range(C.get("n_folds", 5)):
        _, _, te = kfold_split(cases, k, C.get("n_folds", 5), C.get("seed", 42), C.get("val_frac", 0.1))
        for c in te: fold_of[os.path.basename(c)] = k
    rows = []
    for cd in cases:
        img, lab, meta = load_case(cd); name = os.path.basename(cd)
        sp = meta.get("spacing") or meta.get("orig_spacing") or [4.0, 0.73046875, 0.73046875]; sz, sy, sx = float(sp[0]), float(sp[1]), float(sp[2]); vox = sz * sy * sx / 1000.0
        r = {"case": name, "n_slices": int(img.shape[0]), "spacing_z": sz, "spacing_y": sy, "spacing_x": sx, "spacing_source": meta.get("spacing_source", "?"), "voxel_ml": vox, "liver_ml": float((lab > 0).sum() * vox)}
        for c, n in enumerate(SEG_NAMES, start=1): r[f"seg{n}_ml"] = float((lab == c).sum() * vox)
        r["seg_present_count"] = int(sum(r[f"seg{n}_ml"] > 0 for n in SEG_NAMES))
        mp = os.path.join(cd, "mask.npy")
        try:
            m = np.load(mp, mmap_mode="r"); tot = 0.0
            for ch in range(10, min(18, m.shape[0])):
                v = float(np.asarray(m[ch]).sum() * vox); r[f"lesion_ch{ch}_ml"] = v; tot += v
            r["lesion_total_ml"] = tot
        except Exception: r["lesion_total_ml"] = float("nan")
        zs = np.nonzero((lab > 0).any((1, 2)))[0]; r["liver_z0"] = int(zs[0]) if len(zs) else -1; r["liver_z1"] = int(zs[-1]) if len(zs) else -1
        # -- Orientation QC. Convention: z cranial to caudal, y anterior to posterior, x right to left. All three axes are checked against GT anatomy --
        def _cent(m, ax):
            idx = np.nonzero(m)
            return float(np.mean(idx[ax])) if len(idx[0]) else float("nan")
        liv = lab > 0
        ok_x = _cent(liv, 2) < lab.shape[2] * 0.5 + 10          # the liver lies on the patient's right (low x in the image)
        s2, s6 = lab == 2, lab == 7                              # II (superior) vs VI (inferior)
        ok_z = (_cent(s2, 0) < _cent(s6, 0)) if (s2.any() and s6.any()) else True
        s5, s7 = lab == 6, lab == 8                              # V (anterior) vs VII (posterior)
        ok_y = (_cent(s5, 1) < _cent(s7, 1)) if (s5.any() and s7.any()) else True
        r["orient_ok"] = bool(ok_x and ok_y and ok_z); r["orient_x"] = bool(ok_x); r["orient_y"] = bool(ok_y); r["orient_z"] = bool(ok_z)
        r["liver_median_u8"] = float(np.median(img[lab > 0])) if (lab > 0).any() else float("nan"); r["image_mean_u8"] = float(img.mean())
        r["fold_test"] = fold_of.get(name, -1); rows.append(r)
    # Lesion channel mapping priority: (1) data folder (site-edited copy), (2) bundled configs/lesion_labels/<site>.json, (3) template
    import shutil
    lj = os.path.join(a.data, "lesion_labels.json"); bj = os.path.join("configs", "lesion_labels", f"{a.site}.json")
    if a.lesion_labels and os.path.exists(a.lesion_labels):
        shutil.copy(a.lesion_labels, os.path.join(root, "lesion_labels.json")); print(f"Lesion mapping: using file given by argument ({a.lesion_labels})")
    elif os.path.exists(lj):
        shutil.copy(lj, os.path.join(root, "lesion_labels.json")); print(f"Lesion mapping: using file provided in the data folder ({lj})")
    elif os.path.exists(bj):
        shutil.copy(bj, os.path.join(root, "lesion_labels.json")); print(f"Lesion mapping: using bundled mapping for {a.site} ({bj})")
    else:
        print("Lesion mapping: none found; recording by channel number (TEMPLATE created)")
        json.dump({"_description": "Mapping from lesion channels (10-17) to lesion types. Fill in at the site and save as lesion_labels.json in the data folder", "10": "", "11": "", "12": "", "13": "", "14": "", "15": "", "16": "", "17": ""},
                  open(os.path.join(root, "lesion_labels_TEMPLATE.json"), "w"), ensure_ascii=False, indent=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(root, "patient_catalog.csv"), index=False)
    bad = df[~df.orient_ok].case.tolist() if "orient_ok" in df else []
    if bad:
        msg = f"[WARNING] Suspected orientation convention violation in {len(bad)} patients: {bad[:5]}{' ...' if len(bad) > 5 else ''}. Check the axis order of these npy files (cranial->caudal / anterior->posterior / right->left). Flipped inputs invalidate the segmentation."
        print(msg)
        open(os.path.join(root, "ORIENTATION_WARNING.txt"), "w").write(msg + "\n" + "\n".join(df[~df.orient_ok].case.astype(str)))
    print(f"patient_catalog.csv: {len(df)} patients -> {root} | orientation QC: passed {int(df.orient_ok.sum())}/{len(df)}")


if __name__ == "__main__":
    main()
