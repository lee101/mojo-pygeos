"""Compiled-kernel loader and ndarray helpers."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYGEOS_LIB") or os.path.join(ROOT, "dist", "libmojo-pygeos.so")
I, F = ctypes.c_int64, ctypes.c_double
SIGNATURES = {
    "mpg_metrics": ([I] * 6 + [I] * 4, None),
    "mpg_point_relation": ([I] * 8, None),
    "mpg_point_distance_scalar": ([I, I, I, F, F, I], None),
}


def build(force: bool = False) -> str:
    src = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(src):
        return LIB
    if os.environ.get("MOJO_PYGEOS_LIB"):
        raise RuntimeError(f"MOJO_PYGEOS_LIB does not exist or is stale: {LIB}")
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(["bash", script], cwd=ROOT, text=True, capture_output=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _loaded


def f64(values) -> np.ndarray:
    """Return a contiguous float64 array without silently losing precision."""
    raw = np.asarray(values)
    if raw.dtype.kind == "c":
        raise TypeError("complex coordinates are not supported")
    if raw.dtype.kind in "iu" and raw.size:
        limit = 2**53
        if raw.dtype.kind == "u":
            unsafe = np.any(raw > limit)
        else:
            unsafe = np.any(raw > limit) or np.any(raw < -limit)
        if unsafe:
            raise ValueError("integer coordinates must be exactly representable as float64")
    if raw.dtype.kind == "f" and raw.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise ValueError("coordinates with precision wider than float64 are not supported")
    return np.ascontiguousarray(raw, dtype=np.float64)


def i64(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError("offsets and indices must use an integer dtype")
    info = np.iinfo(np.int64)
    if raw.size and (np.any(raw < info.min) or np.any(raw > info.max)):
        raise ValueError("offsets and indices must fit in int64")
    return np.ascontiguousarray(raw, dtype=np.int64)


def addr(values: np.ndarray) -> int:
    return values.ctypes.data
