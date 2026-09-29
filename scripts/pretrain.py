#!/usr/bin/env python
"""Pretraining on public datasets (single 48 GB GPU).

  python scripts/pretrain.py --mode pretrain   # excludes Lee-50 and MedSeg-41; group-wise train/val split, then evaluation on held-out test sets
  python scripts/pretrain.py --mode tian5fold --fold 0   # reproduces the TotalSegmentator 570-case protocol (0.897)
"""
import os, sys, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from couinaudfl.model import build_model, load_ts_init, find_ts_checkpoint
from couinaudfl.train import fit, validate
from couinaudfl.data import load_case
from couinaudfl.infer import predict_volume, predict_volume_auto_orient
from couinaudfl.metrics import case_metrics, summarize
from couinaudfl import splits as S

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/public"); ap.add_argument("--mode", choices=["pretrain", "tian5fold", "mr5fold"], default="pretrain")
ap.add_argument("--fold", type=int, default=0); ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--amp", type=int, default=1); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--val-max", type=int, default=30)
ap.add_argument("--ts", default="checkpoints/totalsegmentator/291", help="TotalSegmentator weight folder ('' = train from scratch)")
ap.add_argument("--out", default=None); ap.add_argument("--limit", type=int, default=0, help="Debug: limit the number of cases")
ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--no-hd", action="store_true")

def main():
    a = ap.parse_args()
    out = a.out or f"outputs/{a.mode}" + (f"_fold{a.fold}" if a.mode != "pretrain" else ""); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    P = lambda rel: os.path.join(a.root, rel)

    if a.mode == "pretrain":
        sp = S.make_pretrain_split(a.root); train, val = sp["train"], sp["val"]
        tests = {"lee50_tian": sp["test_lee50_tian"], "lee50_nih": sp["test_lee50_nih"], "medseg9": sp["test_medseg"], "mr": sp["test_mr"], "crlm_liver": sp["test_crlm"]}
    elif a.mode == "mr5fold":
        mr = [f"06_ts_mr/{c}" for c in sorted(os.listdir(P("06_ts_mr"))) if os.path.isdir(P(f"06_ts_mr/{c}"))]
        import random; random.Random(42).shuffle(mr); folds = [sorted(mr[k::5]) for k in range(5)]; test = folds[a.fold]
        rest = [c for k in range(5) if k != a.fold for c in folds[k]]; val = rest[:max(3, len(rest) // 10)]; train = rest[len(val):]
        tests = {f"mr_fold{a.fold}": test, "ct_lee50_tian": S.make_pretrain_split(a.root)["test_lee50_tian"]}
    else:
        sp = S.make_tian5fold_split(a.root); folds = sp["folds"]; test = folds[a.fold]
        rest = [c for k in range(5) if k != a.fold for c in folds[k]]; val = rest[:max(3, len(rest) // 10)]; train = rest[len(val):]
        tests = {f"tian_fold{a.fold}": test}
    if a.limit: train, val = train[:a.limit], val[:max(2, a.limit // 5)]; tests = {k: v[:a.limit] for k, v in tests.items()}
    json.dump({"train": train, "val": val, "tests": tests}, open(os.path.join(out, "split_used.json"), "w"), indent=1)
    log(f"[{a.mode}] train {len(train)} / val {len(val)} / tests { {k: len(v) for k, v in tests.items()} } | TS init: {a.ts or 'scratch'} | amp={a.amp}")

    dev = torch.device("cuda"); model = build_model()
    if a.ts: st = load_ts_init(model, find_ts_checkpoint(a.ts)); json.dump({k: v for k, v in st.items() if k != "skipped_from_ckpt"}, open(os.path.join(out, "ts_init.json"), "w"), indent=1)
    if not a.eval_only:
        r = fit(model, [P(c) for c in train], [P(c) for c in val], out, dev, epochs=a.epochs, lr=a.lr, amp=bool(a.amp), num_workers=a.workers, val_max=a.val_max, log=log)
        log(f"Training finished, best val {r['best_val_dice']:.4f}")
    bp = os.path.join(out, "best.pth"); model.load_state_dict(torch.load(bp, map_location=dev)); model.eval(); log(f"Evaluation weights: {bp}")

    results = {}
    for name, cases in tests.items():
        rows = []
        for rel in cases:
            img, lab, meta = load_case(P(rel)); pred, _, _fl = predict_volume_auto_orient(model, img, dev, amp=bool(a.amp))
            seg8 = int(meta.get("segments", 9)) == 8 or name.startswith("lee") or name.startswith("tian") or name.startswith("mr")
            if name == "crlm_liver":   # liver-only labels: liver (1) + remnant (2) = whole liver vs. union of predicted segments
                lab = (np.isin(lab, [1, 2])).astype(np.uint8); pred = (pred > 0).astype(np.uint8); seg8 = True
            os_ = meta.get("orig_spacing") or [4.0, 0.7, 0.7]; sp_ = [4.0, float(os_[1]), float(os_[2])]   # z is resampled to 4 mm; in-plane spacing is the original (512 x 512 source)
            rows += case_metrics(pred, lab, sp_, rel.split("/")[-1], seg8=seg8, with_hd=not a.no_hd)
            np.save(os.path.join(out, "pred_" + name + "_" + rel.split("/")[-1] + ".npy"), pred) if len(cases) <= 60 else None
        import pandas as pd; pd.DataFrame(rows).to_csv(os.path.join(out, f"metrics_{name}.csv"), index=False)
        results[name] = summarize(rows); log(f"[{name}] {json.dumps(results[name])}")
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1); log("Done")


if __name__ == "__main__":
    main()