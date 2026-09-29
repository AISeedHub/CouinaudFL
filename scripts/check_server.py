#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-flight server connectivity check (run once at each site **before** run_center).

The FL port is first contacted only after the single-site 5-fold stage finishes, so a blocking firewall
would otherwise be discovered only hours later. This script checks all three gates within 10 seconds.

  Linux : .venv/bin/python scripts/check_server.py
  Windows: .venv\\Scripts\\python.exe scripts\\check_server.py
"""
import os, re, sys, socket, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg_get(path, key):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'^%s:\s*"?([^"#\s]+)"?' % re.escape(key), line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def tcp(host, port, timeout=5):
    try:
        socket.create_connection((host, int(port)), timeout).close()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp4c", help="Experiment names to check (comma-separated, configs/<exp>.yaml)")
    args = ap.parse_args()
    print("=== Server connectivity check ===")
    targets, upload_url, unresolved = [], None, []
    for exp in [e.strip() for e in args.exp.split(",") if e.strip()]:
        p = os.path.join(HERE, "configs", f"{exp}.yaml")
        if not os.path.exists(p):
            continue
        addr = cfg_get(p, "client_server_address") or cfg_get(p, "server_address")
        uurl = cfg_get(p, "upload_url")
        if uurl and "SERVER_IP" not in uurl:
            upload_url = uurl
        if addr:
            host, _, port = addr.rpartition(":")
            # The site-side config must contain the actual address the client will connect to
            if host in ("0.0.0.0", "", "SERVER_IP"):
                print(f"[{exp}] client address={addr}: replace SERVER_IP with the actual server address")
                unresolved.append(exp)
                continue
            targets.append((exp, host, port))

    if not targets and not upload_url:
        print("\nResult: no actual server address in the configs. Replace SERVER_IP in configs/*.yaml and run again.")
        return 2

    ok = not unresolved
    for exp, host, port in targets:
        good, err = tcp(host, port)
        print(f"[{exp}] FL {host}:{port} … {'OK' if good else 'FAIL  ' + err}")
        ok &= good

    if upload_url:
        base = upload_url.rsplit("/", 1)[0]
        host = re.sub(r"^https?://", "", base).split("/")[0]
        h, _, p = host.rpartition(":")
        good, err = tcp(h or host, p or 80)
        print(f"[collect] {host} … {'OK' if good else 'FAIL  ' + err}")
        ok &= good
        if good:  # Also check the weights endpoint (without downloading)
            try:
                req = urllib.request.Request(base + "/weights", headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read(1)
                    print(f"[weights] {base}/weights ... OK")
            except Exception as e:
                print(f"[weights] {base}/weights ... check failed ({type(e).__name__}); the first run will fall back to the GitHub release")

    print("\nResult: " + ("all gates passed; run_center can be started" if ok else
                      "some checks failed; verify the firewall and that the server is running, then re-check (running now would stall at the FL stage after single-site training)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
