#!/usr/bin/env python
"""FL client: automatically reconnects following the same fold x method order as the server. Data are in the institutional folder format (per-patient image.npy/mask.npy).

  python scripts/client.py --config configs/exp4c.yaml --data <data_root> --site A
When the server finishes a session, the client moves on to the next (fold, method) and reconnects (retrying every 30 s on connection failure).
If STOP.txt exists in outputs/<exp>/, the client exits after the current session. Completed (fold, method) pairs are skipped via DONE markers.
"""
import os, sys, time, json, argparse, datetime, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, flwr as fl, signal
# Prevent NAT idle timeouts from dropping the connection during local training (10 epochs, ~13 min), which would lose
# the result upload with GrpcBridgeClosed. A gRPC keepalive ping is sent every 120 s, about half of the ~300 s NAT/firewall idle timeout measured on one site's network path. The server sets max_ping_strikes=0 so pings are never rejected.
import grpc as _grpc
_orig_ich = _grpc.insecure_channel
def _ich_keepalive(target, options=None, compression=None):
    opts = list(options or []) + [("grpc.keepalive_time_ms", 120000), ("grpc.keepalive_timeout_ms", 1200000),
                                  ("grpc.keepalive_permit_without_calls", 1), ("grpc.http2.max_pings_without_data", 0)]
    return _orig_ich(target, options=opts, compression=compression)
_grpc.insecure_channel = _ich_keepalive
# Ignore accidental terminal interrupts (Ctrl+C/Break); stop only via STOP.txt
for _sig in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, _sig): signal.signal(getattr(signal, _sig), signal.SIG_IGN)
if os.name == "nt":
    try:
        import ctypes; k = ctypes.windll.kernel32; h = k.GetStdHandle(-10); m = ctypes.c_uint(); k.GetConsoleMode(h, ctypes.byref(m))
        k.SetConsoleMode(h, m.value & ~0x0040 | 0x0080)   # QuickEdit off, ExtendedFlags on
        k.SetThreadExecutionState(0x80000003)              # prevent sleep and display power-off
    except Exception: pass
from couinaudfl.model import build_model
from couinaudfl.data import list_cases, kfold_split
from couinaudfl.fl import CouinaudClient, METHODS
from couinaudfl.fsutil import write_marker, ensure_free

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--data", required=True); ap.add_argument("--site", default=socket.gethostname())
ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--run", default=None, help="Run name (default: current timestamp). Re-running with the same name resumes the run."); ap.add_argument("--methods", nargs="*", default=None); ap.add_argument("--workers", type=int, default=4)

def main():
    ap.add_argument("--server", default=None, help="override server_address"); ap.add_argument("--spacing-file", default=None, help="Path to the per-patient voxel spacing table (csv/xlsx)"); a = ap.parse_args()
    C = yaml.safe_load(open(a.config, encoding="utf-8")); folds = a.folds if a.folds is not None else C.get("folds", [0, 1, 2, 3, 4]); methods = a.methods or C.get("methods", METHODS)
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
            if os.path.exists(os.path.join(root, "STOP.txt")): log("STOP.txt found, stopping"); sys.exit(0)
            if os.path.exists(done): log(f"skip fold{fold} {method}"); continue
            ensure_free(root, 10, f"before fold{fold} {method}")
            log(f"=== fold {fold} / {method} : train {len(tr)} val {len(va)} test {len(te)} → {server}")
            model = client = None
            while True:
                try:
                    if client is None:   # first attempt or after an exception: rebuild the model and client to reset GPU state
                        model = build_model().to(dev)
                        client = CouinaudClient(model, tr, va, te, dev, od, amp=amp, workers=a.workers, log=log, cid=a.site)
                        client.expected_fold = fold; client.expected_method = method
                    fl.client.start_client(server_address=server, client=client.to_client())
                    # Verify session completion: a closed connection is not completion. Count as done only if the final-round weights were actually saved
                    if os.path.exists(os.path.join(od, method, f"r{C['rounds']:02d}_last.pth")): break
                    log(f"Connection closed but session incomplete (r{C['rounds']:02d} missing); reconnecting in 60 s")
                    import gc as _gc; del client, model; client = model = None; _gc.collect(); torch.cuda.empty_cache(); time.sleep(60)
                except Exception as e:
                    import traceback, gc; tb = traceback.format_exc()
                    if "SESSION_MISMATCH" in tb:
                        log(f"Session mismatch detected; training halted, waiting: {str(e)[:160]}"); time.sleep(60)
                    elif "grpc" in tb.lower() or "connect" in str(e).lower() or "UNAVAILABLE" in tb:
                        log(f"Waiting for connection ({type(e).__name__}); retrying in 30 s"); time.sleep(30)
                    else:   # exception during training: log it, discard and rebuild the model, then reconnect (the server proceeds to the next round after round_timeout)
                        log(f"Training exception logged to {os.path.join(root, 'errors.log')}; rebuilding the model and reconnecting in 30 s"); open(os.path.join(root, "errors.log"), "a").write(f"\n{datetime.datetime.now()} fold{fold} {method}\n{tb}")
                        del client, model; client = model = None; gc.collect(); torch.cuda.empty_cache(); time.sleep(30)
            write_marker(done, str(datetime.datetime.now())); log(f"fold {fold} {method} finished")
            del model, client; torch.cuda.empty_cache()
            # Upload results to the server as soon as each method finishes (intermediate backup; does not wait for the full run).
            # Exceptions are ignored so that upload failures never affect training.
            try:
                import subprocess as _su
                _su.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_results.py"), "--exp", C["experiment"], "--site", a.site, "--run", run], check=False)
                _su.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_results.py"), "--exp", C["experiment"], "--site", a.site, "--run", run], check=False)
                log(f"[auto_upload] fold{fold} {method} results sent to server")
            except Exception as _e:
                log(f"[auto_upload] failed (ignored, continuing): {str(_e)[:120]}")
            time.sleep(C.get("gap_sec", 10))
    log("All sessions finished ALL_SESSIONS_DONE")


if __name__ == "__main__":
    main()