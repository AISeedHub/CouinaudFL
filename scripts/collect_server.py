#!/usr/bin/env python
"""Result collection server (port 9598): receives, stores, and extracts zip archives sent by upload_results.py at each site.

  nohup .venv/bin/python scripts/collect_server.py --port 9598 --dir outputs/collected > outputs/collect.log 2>&1 &
Storage: outputs/collected/<exp>/<site>/<timestamp>/ (original zip + extracted files). Repeated uploads from the same site are kept in new timestamped folders.
"""
import os, argparse, datetime, zipfile, io, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=9598); ap.add_argument("--dir", default="outputs/collected"); a = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            q = parse_qs(urlparse(self.path).query); exp = q.get("exp", ["unknown"])[0]; site = q.get("site", ["unknown"])[0]; sha = q.get("sha", [""])[0]; run = q.get("run", ["norun"])[0]
            n = int(self.headers.get("Content-Length", 0)); data = self.rfile.read(n)
            got = hashlib.sha256(data).hexdigest()[:16]
            if sha and got != sha: self.send_response(400); self.end_headers(); self.wfile.write(b"sha mismatch"); return
            d = os.path.join(a.dir, exp, run, site, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")); os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f"export_{site}.zip"), "wb").write(data)
            try: zipfile.ZipFile(io.BytesIO(data)).extractall(d); msg = f"saved {n/2**20:.1f}MB → {d}"
            except zipfile.BadZipFile: msg = f"saved raw (bad zip) → {d}"
            print(datetime.datetime.now(), exp, site, msg, flush=True)
            self.send_response(200); self.end_headers(); self.wfile.write(msg.encode())

        def do_GET(self):
            if urlparse(self.path).path == "/weights":   # Distribute pretrained weights
                wp = "outputs/init_weights.pth" if os.path.exists("outputs/init_weights.pth") else "outputs/pretrain/best.pth"
                if os.path.exists(wp):
                    self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(os.path.getsize(wp))); self.end_headers()
                    with open(wp, "rb") as f:
                        while True:
                            b = f.read(1 << 20)
                            if not b: break
                            self.wfile.write(b)
                else: self.send_response(404); self.end_headers(); self.wfile.write(b"weights not found")
                return
            self.send_response(200); self.end_headers(); self.wfile.write(b"CouinaudFL collect server OK")

        def log_message(self, *args): pass

    os.makedirs(a.dir, exist_ok=True); print(f"collect server on :{a.port} → {a.dir}", flush=True)
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
