#!/usr/bin/env python
"""FL client. Iterates over (fold, method) sessions in the same order as the server and reconnects automatically. Data: institutional-format folder (per-patient image.npy/mask.npy).

  python scripts/client.py --config configs/exp4c.yaml --data <data_root> --site A
When the server finishes a session, the client moves to the next (fold, method) and reconnects (retrying every 30 s on connection failure).
If outputs/<exp>/STOP.txt exists, the client exits after the current session. Completed (fold, method) pairs are skipped via DONE markers.
"""
import os, sys, time, json, argparse, datetime, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, flwr as fl, signal
# Ignore accidental Ctrl+C/Break in the terminal; stop only via STOP.txt
for _sig in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, _sig): signal.signal(getattr(signal, _sig), signal.SIG_IGN)
if os.name == "nt":
    try:
        import ctypes; k = ctypes.windll.kernel32; h = k.GetStdHandle(-10); m = ctypes.c_uint(); k.GetConsoleMode(h, ctypes.byref(m))
        k.SetConsoleMode(h, m.value & ~0x0040 | 0x0080)   # QuickEdit off, ExtendedFlags on
        k.SetThreadExecutionState(0x80000003)              # prevent system sleep / display off
    except Exception: pass
from couinaudfl.model import build_model
from couinaudfl.data import list_cases, kfold_split
from couinaudfl.fl import CouinaudClient, METHODS

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--data", required=True); ap.add_argument("--site", default=socket.gethostname())
ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--run", default=None, help="run name (default: current timestamp); rerunning with the same name resumes"); ap.add_argument("--methods", nargs="*", default=None); ap.add_argument("--workers", type=int, default=4)

def main():
    ap.add_argument("--server", default=None, help="override server_address"); ap.add_argument("--spacing-file", default=None, help="path to per-patient spacing table (csv/xlsx)"); a = ap.parse_args()
    C = yaml.safe_load(open(a.config)); folds = a.folds if a.folds is not None else C.get("folds", [0, 1, 2, 3, 4]); methods = a.methods or C.get("methods", METHODS)
    from couinaudfl.data import set_spacing_file
    if a.spacing_file: set_spacing_file(a.spacing_file)
    server = a.server or C["client_server_address"]; run = a.run or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"); root = os.path.join(C.get("output_dir", "outputs"), C["experiment"], run, f"client_{a.site}"); os.makedirs(root, exist_ok=True)
    logf = open(os.path.join(root, "client.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); cases = list_cases(a.data)
    log(f"[{C['experiment']}] site {a.site} | {len(cases)} patients | server {server} | device {dev} {torch.cuda.get_device_name(0) if dev.type=='cuda' else ''}")
    amp = bool(C.get("amp", 0))
    for fold in folds:
        tr, va, te = kfold_split(cases, fold, C.get("n_folds", 5), C.get("seed", 42), C.get("val_frac", 0.1))
        json.dump({"train": [os.path.basename(x) for x in tr], "val": [os.path.basename(x) for x in va], "test": [os.path.basename(x) for x in te]},
                  open(os.path.join(root, f"fold{fold}_split.json"), "w"), indent=1)
        for method in methods:
            od = os.path.join(root, f"fold{fold}"); done = os.path.join(od, f"DONE_{method}")
            if os.path.exists(os.path.join(root, "STOP.txt")): log("STOP.txt found, exiting"); sys.exit(0)
            if os.path.exists(done): log(f"skip fold{fold} {method}"); continue
            log(f"=== fold {fold} / {method} : train {len(tr)} val {len(va)} test {len(te)} → {server}")
            model = client = None
            while True:
                try:
                    if client is None:   # first time or after an exception: rebuild model/client for a clean GPU state
                        model = build_model().to(dev)
                        client = CouinaudClient(model, tr, va, te, dev, od, amp=amp, workers=a.workers, log=log, cid=a.site)
                    fl.client.start_client(server_address=server, client=client.to_client()); break
                except Exception as e:
                    import traceback, gc; tb = traceback.format_exc()
                    if "grpc" in tb.lower() or "connect" in str(e).lower() or "UNAVAILABLE" in tb:
                        log(f"Waiting for server ({type(e).__name__}); retrying in 30 s"); time.sleep(30)
                    else:   # exception during training: log it, discard the model, rebuild and reconnect (the server proceeds after round_timeout)
                        log(f"Training exception logged to {os.path.join(root, 'errors.log')}; rebuilding model and reconnecting in 30 s"); open(os.path.join(root, "errors.log"), "a").write(f"\n{datetime.datetime.now()} fold{fold} {method}\n{tb}")
                        del client, model; client = model = None; gc.collect(); torch.cuda.empty_cache(); time.sleep(30)
            open(done, "w").write(str(datetime.datetime.now())); log(f"fold {fold} {method} finished")
            del model, client; torch.cuda.empty_cache(); time.sleep(C.get("gap_sec", 10))
    log("All sessions finished ALL_SESSIONS_DONE")


if __name__ == "__main__":
    main()