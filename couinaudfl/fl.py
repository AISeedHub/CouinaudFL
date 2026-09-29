"""Federated learning: Flower 1.11 NumPyClient and server strategy.

Methods: FedAvg, FedProx (mu), FedAdam (server-side Adam, Reddi et al. 2021), FedBN-IN (keeps InstanceNorm affine parameters local; a FedBN variant for networks without BatchNorm)
Client design
  * Each round, the global parameters received from the server are snapshotted in `last_global`; evaluate and final testing always use the global model.
  * The server config (method, lr, local_epochs, total_epochs, fold) is followed as is.
  * Training outputs and test metrics are saved locally under outputs/<exp>/fold<k>/<method>/. Case folder names are kept locally and replaced by Case N in exported results.
"""
from __future__ import annotations
import os, json, copy
from collections import OrderedDict
import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.strategy import FedAvg
from .train import fit, validate
from .data import load_case
from .infer import predict_volume, predict_volume_auto_orient
from .metrics import case_metrics, summarize

METHODS = ["FedAvg", "FedProx", "FedAdam", "FedBN"]


def get_nd(model): return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_nd(model, nds, keep_keys=()):
    sd = model.state_dict(); new = OrderedDict()
    for (k, old), v in zip(sd.items(), nds):
        new[k] = old if any(k.endswith(s) or s in k for s in keep_keys) and keep_keys else torch.from_numpy(np.asarray(v)).to(old.dtype)
    model.load_state_dict(new)


def bn_keys(model):
    """FedBN-IN: InstanceNorm affine (weight/bias) keys kept local."""
    return [k for k, m in model.named_modules() if isinstance(m, torch.nn.InstanceNorm3d)]


class CouinaudClient(fl.client.NumPyClient):
    def __init__(self, model, train_cases, val_cases, test_cases, device, out_dir, amp=False, workers=4, log=print, cid="client"):
        self.model, self.device, self.out, self.log, self.cid = model, device, out_dir, log, cid
        self.train_cases, self.val_cases, self.test_cases = train_cases, val_cases, test_cases
        self.amp, self.workers = amp, workers; self.last_global = None; self.round = 0
        self.in_keys = bn_keys(model); os.makedirs(out_dir, exist_ok=True)

    def get_parameters(self, config): return get_nd(self.model)

    expected_fold = None; expected_method = None   # set by client.py at session start (None disables the check)

    def _check_session(self, config, method):
        """Check the server's (fold, method) against the expected local session; abort immediately on mismatch to avoid mixing results."""
        if self.expected_fold is None: return
        sf = int(config.get("fold", -1))
        if sf != int(self.expected_fold) or method != self.expected_method:
            raise RuntimeError(f"SESSION_MISMATCH: server fold{sf}/{method} vs local fold{self.expected_fold}/{self.expected_method}; server and client runs are out of sync. Restart both with a new run.")

    def _apply_global(self, params, method):
        keep = self.in_keys if method == "FedBN" else ()
        set_nd(self.model, params, keep); self.last_global = copy.deepcopy(self.model.state_dict())

    def fit(self, parameters, config):
        method = str(config.get("method", "FedAvg")); rnd = int(config.get("server_round", 1)); self.round = rnd
        self._check_session(config, method)
        le = int(config.get("local_epochs", 10)); total = int(config.get("total_epochs", 200)); lr = float(config.get("lr", 0.01))
        self._apply_global(parameters, method)
        prox_mu = float(config.get("fedprox_mu", 0.0)) if method == "FedProx" else 0.0
        if prox_mu > 0: _install_prox(self.model, prox_mu)
        r = fit(self.model, self.train_cases, None, os.path.join(self.out, method), self.device, epochs=le, lr=lr, amp=self.amp,
                num_workers=self.workers, resume=False, log=self.log, epoch_offset=(rnd - 1) * le, total_epochs=total, save_prefix=f"r{rnd:02d}_")
        if prox_mu > 0: _remove_prox(self.model)
        return get_nd(self.model), len(self.train_cases), {"loss": float(r["history"][-1]["loss"]) if r["history"] else 0.0, "cid": self.cid}

    def evaluate(self, parameters, config):
        method = str(config.get("method", "FedAvg")); rnd = int(config.get("server_round", 0)); final = bool(config.get("final", False))
        self._check_session(config, method)
        self._apply_global(parameters, method)
        v = validate(self.model, self.val_cases, self.device, self.amp)
        os.makedirs(os.path.join(self.out, method), exist_ok=True)   # avoid a missing-folder crash when evaluate arrives before fit (e.g., right after reconnecting)
        torch.save(self.last_global, os.path.join(self.out, method, f"global_r{rnd:02d}.pth"))
        rec = {"round": rnd, "val_dice": v["dice"], "cid": self.cid}
        if final or rnd == int(config.get("num_rounds", -1)):
            rec.update(self.final_test(method))
        self.log(json.dumps(rec))
        with open(os.path.join(self.out, method, "rounds.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
        return 1.0 - v["dice"], len(self.val_cases), rec

    @torch.no_grad()
    def final_test(self, method):
        """Test with the global weights received from the server (last_global), never the locally trained state."""
        from .lesion import lesion_overlap_rows
        self.model.load_state_dict(self.last_global); self.model.eval(); rows = []; lrows = []
        for cd in self.test_cases:
            img, lab, meta = load_case(cd); pred, _, _fl = predict_volume_auto_orient(self.model, img, self.device, amp=self.amp)
            sp = meta.get("spacing") or [4.0, 0.7, 0.7]
            rows += case_metrics(pred, lab, [float(sp[0]), float(sp[1]), float(sp[2])], os.path.basename(cd))
            lrows += lesion_overlap_rows(os.path.basename(cd), cd, lab, pred, [float(sp[0]), float(sp[1]), float(sp[2])])
            pdir = os.path.join(self.out, method, "pred"); os.makedirs(pdir, exist_ok=True)
            np.savez_compressed(os.path.join(pdir, f"{os.path.basename(cd)}.npz"), pred=pred)   # raw label map, so any metric can be recomputed later
        import pandas as pd; pd.DataFrame(rows).to_csv(os.path.join(self.out, method, "test_metrics.csv"), index=False)
        if lrows: pd.DataFrame(lrows).to_csv(os.path.join(self.out, method, "lesion_overlap.csv"), index=False)
        torch.save(self.last_global, os.path.join(self.out, method, "global_final.pth"))
        s = summarize(rows); json.dump(s, open(os.path.join(self.out, method, "test_summary.json"), "w"), indent=1)
        return {"test_" + k: v for k, v in s.items() if not isinstance(v, dict)}


# ── FedProx: hook adding the gradient of (mu/2)||w-w_g||^2 before each optimizer step ──
def _install_prox(model, mu):
    model._prox_ref = [p.detach().clone() for p in model.parameters()]; model._prox_mu = mu
    model._prox_handles = [p.register_hook(lambda g, p=p, r=r: g + mu * (p.detach() - r)) for p, r in zip(model.parameters(), model._prox_ref)]

def _remove_prox(model):
    for h in getattr(model, "_prox_handles", []): h.remove()
    model._prox_handles = []; model._prox_ref = None


class CouinaudStrategy(FedAvg):
    """Extends FedAvg. FedAdam applies the aggregated delta with server-side Adam. Logs rounds and saves the global model."""

    def __init__(self, method, model_keys, out_dir, num_rounds, local_epochs, lr=0.01, fedprox_mu=0.01, server_lr=1e-3, fold=0, round_offset=0, **kw):
        # round_offset: number of rounds already aggregated when resuming an invalidated session. Flower counts 1..remaining,
        # but clients and saved files use the absolute round (offset+r), so an invalidation costs one round rather than the whole session.
        super().__init__(**kw); self.method, self.keys, self.out, self.num_rounds = method, model_keys, out_dir, num_rounds
        self.round_offset = int(round_offset)
        self.local_epochs, self.lr, self.mu, self.server_lr, self.fold = local_epochs, lr, fedprox_mu, server_lr, fold
        self.history = []; self.last_parameters = None; self.m = None; self.v = None; self.t = 0
        self._init_params = kw.get("initial_parameters")   # always restart sessions from the clean pretrained init (Flower discards it after first use by default)
        os.makedirs(out_dir, exist_ok=True)

    def initialize_parameters(self, client_manager):
        return self._init_params

    def aggregate_fit(self, server_round, results, failures):
        # No partial aggregation: if fewer results than min_fit_clients arrive, refuse to aggregate and invalidate the session.
        # (Flower's default can proceed when a dead connection lingers in the client pool, e.g. 4 sampled → 3 results,
        #  which would violate the full-participation multi-site design.)
        # Always log participants and failures to identify the failing client (cid = gRPC peer address)
        ar = server_round + self.round_offset
        for cp, fr in results:
            print(f"[participation] round {ar} OK cid={cp.cid} site={fr.metrics.get('cid', fr.metrics.get('site','?')) if fr.metrics else '?'}", flush=True)
        for f in failures:
            if isinstance(f, tuple):
                print(f"[participation] round {ar} FAIL cid={f[0].cid} res={f[1]!r}"[:500], flush=True)
            else:
                print(f"[participation] round {ar} FAIL exc={f!r}"[:500], flush=True)
        if len(results) < self.min_fit_clients:
            self.history.append({"round": ar, "n_clients": len(results), "failures": len(failures), "invalid": True})
            raise RuntimeError(f"SESSION_INVALID: round {ar} results {len(results)} < required {self.min_fit_clients}")
        return self._aggregate_fit_impl(server_round, results, failures)

    def _cfg(self, rnd, final=False):
        return {"method": self.method, "server_round": rnd + self.round_offset, "local_epochs": self.local_epochs, "total_epochs": self.num_rounds * self.local_epochs,
                "lr": self.lr, "fedprox_mu": self.mu, "num_rounds": self.num_rounds, "fold": self.fold, "final": final}

    def configure_fit(self, server_round, parameters, client_manager):
        self.on_fit_config_fn = lambda r: self._cfg(r); return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        self.on_evaluate_config_fn = lambda r: self._cfg(r, final=(r + self.round_offset == self.num_rounds)); return super().configure_evaluate(server_round, parameters, client_manager)

    def _aggregate_fit_impl(self, server_round, results, failures):
        agg, metrics = super().aggregate_fit(server_round, results, failures)
        if agg is None: return agg, metrics
        if self.method == "FedAdam" and self.last_parameters is not None:
            prev = parameters_to_ndarrays(self.last_parameters); new = parameters_to_ndarrays(agg)
            delta = [n - p for n, p in zip(new, prev)]; self.t += 1; b1, b2, eps = 0.9, 0.99, 1e-3
            self.m = [b1 * m + (1 - b1) * d for m, d in zip(self.m or [np.zeros_like(d) for d in delta], delta)]
            self.v = [b2 * v + (1 - b2) * d * d for v, d in zip(self.v or [np.zeros_like(d) for d in delta], delta)]
            upd = [p + self.server_lr * m / (np.sqrt(v) + eps) for p, m, v in zip(prev, self.m, self.v)]
            agg = ndarrays_to_parameters(upd)
        self.last_parameters = agg
        self._save_global(server_round + self.round_offset)
        self.history.append({"round": server_round, "n_clients": len(results), "failures": len(failures),
                             "client_loss": {r.metrics.get("cid", c.cid): r.metrics.get("loss") for c, r in results}})
        return agg, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        rec = {"round": server_round + self.round_offset, "clients": {r.metrics.get("cid", c.cid): {k: v for k, v in r.metrics.items() if k != "cid"} for c, r in results}}
        self.history.append(rec); json.dump(self.history, open(os.path.join(self.out, f"round_history_{self.method}.json"), "w"), indent=1)
        return loss, metrics

    def _save_global(self, rnd):
        nds = parameters_to_ndarrays(self.last_parameters)
        sd = OrderedDict((k, torch.from_numpy(np.asarray(v))) for k, v in zip(self.keys, nds))
        torch.save(sd, os.path.join(self.out, f"global_{self.method}_r{rnd:02d}.pth"))
        if rnd == self.num_rounds: torch.save(sd, os.path.join(self.out, f"global_{self.method}.pth"))
