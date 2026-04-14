from __future__ import annotations

"""Thin wrapper layer for optional CUDA PQ crypto backends.

This module intentionally stays lightweight: it only resolves a CUDA-capable
backend when one is installed, and otherwise leaves the CPU path untouched.

External adapters can be connected by setting:
- SECAGG_CUDA_KEM_MODULE
- SECAGG_CUDA_SIG_MODULE

If you later package a Python wrapper around cuDilithium / pqc-cuda-kyber,
this module is the place to expose a stable repo-local API without changing
the benchmark callers.
"""

from typing import Any

from .crypto_backend_plugins import load_cuda_kem_adapter, load_cuda_sig_adapter


def load_kem(level: str) -> Any | None:
    loaded = load_cuda_kem_adapter(level)
    if loaded is None:
        return None
    _, adapter = loaded
    return adapter


def load_signature(level: str) -> Any | None:
    loaded = load_cuda_sig_adapter(level)
    if loaded is None:
        return None
    _, adapter = loaded
    return adapter
