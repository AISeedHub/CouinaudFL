#!/usr/bin/env python
"""Zip the anonymized export folder (export_<site>) and send it to the server-side collector (collect_server.py).
The export folder contains no real patient IDs (export_results.py replaces them with Case N; _id_map_LOCAL_ONLY.json exists only in the client_ folder).

  python scripts/upload_results.py --exp exp4c --site A [--url http://SERVER_IP:9598/upload]
On failure (e.g., blocked network), the zip remains at outputs/<exp>/export_<site>.zip and can be transferred manually.
"""
import os, sys, argparse, shutil, urllib.request, json, datetime, hashlib


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--site", required=True)
    ap.add_argument("--url", default=None, help="Default: upload_url in configs/<exp>.yaml, otherwise http://SERVER_IP:9598/upload"); ap.add_argument("--out", default="outputs"); ap.add_argument("--run", required=True); a = ap.parse_args()
    if not a.url:
        cfg = f"configs/{a.exp}.yaml"; url = None
        if os.path.exists(cfg):
            for line in open(cfg, encoding="utf-8"):
                if line.startswith("upload_url:"): url = line.split(":", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
        if not url: print("upload_url is not set: replace SERVER_IP in configs/<exp>.yaml with the actual server address"); sys.exit(2)
        a.url = url
    ex = os.path.join(a.out, a.exp, a.run, f"export_{a.site}")
    if not os.path.isdir(ex): print("Export folder not found: run export_results.py first"); sys.exit(1)
    assert not os.path.exists(os.path.join(ex, "_id_map_LOCAL_ONLY.json")), "ID mapping file found in the export folder: upload aborted"
    zp = shutil.make_archive(os.path.join(a.out, a.exp, a.run, f"export_{a.site}"), "zip", ex); size = os.path.getsize(zp)
    sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()[:16]
    req = urllib.request.Request(a.url + f"?exp={a.exp}&site={a.site}&run={a.run}&sha={sha}", data=open(zp, "rb").read(), method="POST",
                                 headers={"Content-Type": "application/zip", "Content-Length": str(size)})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r: print(f"Upload complete {size/2**20:.1f} MB sha {sha} → {r.read().decode()[:200]}")
    except Exception as e:
        print(f"Upload failed ({type(e).__name__}: {e}). The zip is at {zp}; please transfer it manually."); sys.exit(2)


if __name__ == "__main__":
    main()
