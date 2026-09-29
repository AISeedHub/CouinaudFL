#!/usr/bin/env python
"""FL server: runs every (fold, strategy) session of the given config(s) in order. Multiple configs run sequentially, each on its own port.

  python scripts/server.py --config configs/exp4c.yaml --run run_main   # four-site experiment plus the result collector (9598)
  python scripts/server.py --config configs/exp4c.yaml [--folds 0 1 2 3 4] [--methods FedAvg FedProx FedAdam FedBN]
For each (fold, method) session the server starts a fresh Flower server, initialized with the pretrained weights (init_weights).
"""
import os, sys, json, glob, argparse, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, flwr as fl
# Accept client keepalive pings without rejecting them (GOAWAY too_many_pings)
import grpc as _grpc
_orig_srv = _grpc.server
def _srv_keepalive(*a, **k):
    # Deduplicate keys via a dict so our values take precedence (instead of relying on gRPC duplicate-argument precedence)
    merged = dict(k.get("options") or [])
    merged.update({"grpc.keepalive_permit_without_calls": 1,
                   "grpc.http2.min_recv_ping_interval_without_data_ms": 30000,
                   "grpc.http2.max_pings_without_data": 0,
                   "grpc.http2.max_ping_strikes": 0,
                   "grpc.keepalive_time_ms": 7200000,      # server-side pings effectively disabled (client pings keep NAT alive)
                   "grpc.keepalive_timeout_ms": 1200000})  # avoid premature disconnects during slow uploads (20 min)
    k["options"] = list(merged.items())
    return _orig_srv(*a, **k)
_grpc.server = _srv_keepalive
from flwr.common import ndarrays_to_parameters
from couinaudfl.model import build_model
from couinaudfl.fsutil import write_marker, ensure_free
from couinaudfl.fl import CouinaudStrategy, get_nd, METHODS

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True, nargs="+", help="One or more configs, run sequentially"); ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--collect-port", type=int, default=9598, help="Result collection server port (0 = disabled)"); ap.add_argument("--run", default=None, help="Run name (default: current timestamp). Re-running with the same name resumes the run.")
ap.add_argument("--quiet-min", type=float, default=30, help="exit after all sites uploaded and no new upload for this many minutes")
ap.add_argument("--max-wait-h", type=float, default=24, help="exit anyway after this many hours of waiting for uploads")
ap.add_argument("--upload-margin-min", type=float, default=60, help="count uploads saved up to this many minutes before the last session ended")

def start_collect(port, root):
    """Start the result collection server as a thread in the same process (FL server and collector in one invocation)."""
    import threading, subprocess, sys as _sys
    os.makedirs(root, exist_ok=True)
    proc = subprocess.Popen([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_server.py"), "--port", str(port), "--dir", root],
                            stdout=open(os.path.join(root, "collect.log"), "a"), stderr=subprocess.STDOUT)
    return proc


def _recent_uploads(exp, since):
    """Site upload folders saved at or after `since`.
    The collector stores uploads as outputs/collected/<exp>/<site run>/<SITE>/<timestamp>/, and each site
    keeps its own run name, so we scan every run folder instead of the server's run name."""
    base = os.path.join("outputs", "collected", exp); found = {}
    if not os.path.isdir(base): return found
    for site_run in os.listdir(base):
        rdir = os.path.join(base, site_run)
        for site in (os.listdir(rdir) if os.path.isdir(rdir) else []):
            sdir = os.path.join(rdir, site)
            for stamp in (os.listdir(sdir) if os.path.isdir(sdir) else []):
                p = os.path.join(sdir, stamp)
                if os.path.isdir(p) and os.path.getmtime(p) >= since:
                    found[site] = max(found.get(site, 0.0), os.path.getmtime(p))
    return found


def wait_for_uploads(a, run):
    """After the last session, keep the collector alive until every site has uploaded, then exit.

    Sites upload after the server finishes, so the server cannot stop immediately. It exits when
    (1) at least min_clients sites of the last experiment have uploaded since the sessions ended and
    no new upload has arrived for --quiet-min minutes (late duplicate uploads are allowed to land), or
    (2) --max-wait-h hours have passed, or (3) outputs/STOP_SERVER.txt exists (manual stop)."""
    import yaml as _y, subprocess as _sp
    last = _y.safe_load(open(a.config[-1], encoding="utf-8")); exp_l = last["experiment"]; need = int(last["min_clients"])
    t_done = time.time(); since = t_done - a.upload_margin_min * 60
    print(f"All experiments finished. Collector (:{a.collect_port}) waits for {need} {exp_l} uploads, "
          f"then exits after {a.quiet_min} quiet minutes (max {a.max_wait_h} h). Manual stop: outputs/STOP_SERVER.txt", flush=True)
    analyzed = False
    while True:
        if os.path.exists("outputs/STOP_SERVER.txt"):
            print("STOP_SERVER.txt found. Exiting.", flush=True); return
        if time.time() - t_done > a.max_wait_h * 3600:
            print(f"Max wait {a.max_wait_h} h reached. Exiting.", flush=True); return
        got = _recent_uploads(exp_l, since)
        if len(got) >= need:
            if not analyzed:
                time.sleep(60)   # let the last zip finish extracting
                roots = []   # latest upload per site only, so repeated uploads are not counted twice
                for c in a.config:
                    base = os.path.join("outputs", "collected", _y.safe_load(open(c, encoding="utf-8"))["experiment"]); latest = {}
                    for p in glob.glob(os.path.join(base, "*", "*", "*")):
                        if os.path.isdir(p):
                            site = os.path.basename(os.path.dirname(p))
                            if site not in latest or os.path.getmtime(p) > os.path.getmtime(latest[site]): latest[site] = p
                    roots += [latest[s] for s in sorted(latest)]
                outd = os.path.join("outputs", "analysis_lesion", run)
                r = _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_lesion_roc.py"),
                             "--roots", *roots, "--out", outd], capture_output=True, text=True)
                print(f"[auto analysis] lesion ROC -> {outd} (exit {r.returncode})", flush=True)
                if r.returncode != 0: print(r.stderr[-500:], flush=True)
                analyzed = True
            quiet = time.time() - max(got.values())
            if quiet >= a.quiet_min * 60:
                print(f"All {len(got)} sites uploaded ({', '.join(sorted(got))}); no new upload for {a.quiet_min} min. Exiting.", flush=True)
                return
        time.sleep(30)


def main():
    import signal, sys as _s
    signal.signal(signal.SIGTERM, lambda *a: _s.exit(0))   # ensure the finally block (collector shutdown) runs on kill
    ap.add_argument("--methods", nargs="*", default=None); a = ap.parse_args()
    run = a.run or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    collect = start_collect(a.collect_port, "outputs/collected") if a.collect_port else None
    try:
        for cfg in a.config: run_experiment(cfg, a, run)
        if collect:
            wait_for_uploads(a, run)
    finally:
        if collect: collect.terminate()


def run_experiment(cfg_path, a, run):
    C = yaml.safe_load(open(cfg_path, encoding="utf-8")); folds = a.folds if a.folds is not None else C.get("folds", [0, 1, 2, 3, 4]); methods = a.methods or C.get("methods", METHODS)
    out = os.path.join(C.get("output_dir", "outputs"), C["experiment"], run, "server"); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "server.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()

    model = build_model(); keys = list(model.state_dict().keys())
    if C.get("init_weights"):
        sd = torch.load(C["init_weights"], map_location="cpu", weights_only=False); sd = sd.get("model", sd); model.load_state_dict(sd); log(f"init: {C['init_weights']}")
    init_nd = get_nd(model); del model
    log(f"run={run}"); log(f"[{C['experiment']}] address {C['server_address']} | min_clients {C['min_clients']} | rounds {C['rounds']} x local {C['local_epochs']} | folds {folds} | methods {methods}")
    for fold in folds:
        for method in methods:
            done = os.path.join(out, f"fold{fold}", f"DONE_{method}")
            if os.path.exists(done): log(f"skip fold{fold} {method} (done)"); continue
            od = os.path.join(out, f"fold{fold}"); log(f"=== fold {fold} / {method} : waiting for {C['min_clients']} clients ===")
            # Round-level resume: if an aggregated global_<method>_rXX.pth exists, continue from it (remaining rounds only).
            # An invalid session (insufficient participation) then loses only the last unaggregated round, not the whole session.
            import re as _re, glob as _glob
            _gl = sorted(_glob.glob(os.path.join(od, f"global_{method}_r*.pth")))
            _resume_r = max([int(_re.search(r"_r(\d+)\.pth$", g).group(1)) for g in _gl], default=0)
            _resume_r = min(_resume_r, C["rounds"] - 1) if _resume_r else 0   # if the final round is complete, the normal path writes DONE
            if _resume_r > 0:
                _sd = torch.load(os.path.join(od, f"global_{method}_r{_resume_r:02d}.pth"), map_location="cpu", weights_only=False)
                _m = build_model(); _m.load_state_dict(_sd); _sess_init = get_nd(_m); del _m
                log(f"[resume] fold{fold} {method}: aggregated weights found up to round {_resume_r}; running {C['rounds']-_resume_r} rounds from round {_resume_r+1} (FedAdam moments are reset)")
            else:
                _sess_init = init_nd
            strat = CouinaudStrategy(method, keys, od, C["rounds"], C["local_epochs"], lr=C.get("lr", 0.01), fedprox_mu=C.get("fedprox_mu", 0.01),
                                     server_lr=C.get("fedadam_lr", 1e-3), fold=fold, round_offset=_resume_r, fraction_fit=1.0, fraction_evaluate=1.0,
                                     min_fit_clients=C["min_clients"], min_evaluate_clients=C["min_clients"], min_available_clients=C["min_clients"],
                                     initial_parameters=ndarrays_to_parameters(_sess_init))
            while True:
                ensure_free(out, 5, f"before fold{fold} {method} session")
                t0 = time.time()
                try:
                    fl.server.start_server(server_address=C["server_address"], config=fl.server.ServerConfig(num_rounds=C["rounds"] - _resume_r, round_timeout=float(C.get("round_timeout_sec", 10800))), strategy=strat)
                except RuntimeError as e:
                    if "SESSION_INVALID" in str(e):
                        # Flower does not clean up the gRPC listener on the exception path, so restarting within the same process
                        # fails with 'Port already in use'. The whole process re-executes itself to release the port and stale state.
                        # Execution resumes automatically from the first session without DONE (i.e., this one).
                        log(f"[WARNING] fold {fold} {method}: session invalid due to insufficient participation; restarting the server process in 60 s ({e})")
                        logf.flush(); time.sleep(60)
                        import subprocess as _sp, sys as _sys
                        _sp.run(["pkill", "-9", "-f", "collect_server"], check=False)
                        os.execv(_sys.executable, [_sys.executable] + _sys.argv)
                    raise
                got = sum(int(h["n_clients"]) for h in strat.history if "n_clients" in h)
                # Check fit records only: history entries written by aggregate_evaluate have no n_clients key (read as 0),
                # which previously caused completed sessions to be misclassified as having insufficient participation.
                fit_hist = [h for h in strat.history if "n_clients" in h]
                bad = any(h.get("invalid") for h in fit_hist) or any(int(h["n_clients"]) < C["min_clients"] for h in fit_hist)
                if got == 0 or bad:
                    log(f"[WARNING] fold {fold} {method}: insufficient valid results (all failed or under-participated rounds); restarting the same session in 60 s without writing DONE")
                    time.sleep(60); strat.history = []; strat.last_parameters = None; continue
                break
            write_marker(done, f"{datetime.datetime.now()} {time.time()-t0:.0f}s\n"); log(f"fold {fold} {method} finished {time.time()-t0:.0f}s")
            time.sleep(C.get("gap_sec", 10))
    log("All sessions finished"); logf.close()


if __name__ == "__main__":
    main()