# -*- coding: utf-8 -*-
"""Filesystem safety utilities that prevent silent corruption when the disk is full.
write_marker: atomically writes a DONE marker (temp file → fsync → rename) and verifies its content.
check_free_gb: free space (GB) at a path. ensure_free: raises a clear error if below the threshold."""
from __future__ import annotations
import os, shutil, tempfile


def write_marker(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".marker_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        if os.path.getsize(tmp) == 0:
            raise IOError(f"marker write failed (0 bytes): {path}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"marker verification failed: {path}")


def check_free_gb(path: str) -> float:
    return shutil.disk_usage(os.path.abspath(path)).free / 1e9


def ensure_free(path: str, min_gb: float, what: str = "") -> None:
    free = check_free_gb(path)
    if free < min_gb:
        raise RuntimeError(f"DISK_LOW {what}: free {free:.1f} GB < required {min_gb} GB ({path}); free up disk space and rerun")
