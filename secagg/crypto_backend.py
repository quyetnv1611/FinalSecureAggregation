from __future__ import annotations

import importlib.util
import logging
import os
from typing import Literal, Tuple

logger = logging.getLogger(__name__)

AccelMode = Literal["cpu", "cuda"]


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_requested_mode(explicit: str | None = None) -> str:
    """Return requested acceleration mode: cpu, cuda, or auto.

    Precedence:
    1) explicit argument
    2) SECAGG_CRYPTO_ACCEL env var
    3) auto
    """
    raw = (explicit or os.getenv("SECAGG_CRYPTO_ACCEL", "auto")).strip().lower()
    if raw not in {"cpu", "cuda", "auto"}:
        logger.warning(
            "Unknown SECAGG_CRYPTO_ACCEL=%r, falling back to 'auto'.",
            raw,
        )
        return "auto"
    return raw


def resolve_mode(explicit: str | None = None) -> AccelMode:
    requested = get_requested_mode(explicit)
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if _torch_cuda_available() else "cpu"
    return "cuda" if _torch_cuda_available() else "cpu"


def split_backend_qualifier(backend: str) -> Tuple[str, str | None]:
    """Split backend qualifier syntax: 'NAME@cuda' or 'NAME@cpu'."""
    if "@" not in backend:
        return backend, None
    base, mode = backend.rsplit("@", 1)
    mode = mode.strip().lower()
    if mode not in {"cpu", "cuda", "auto"}:
        return backend, None
    return base, mode


def cuda_kem_available() -> bool:
    # Placeholder probe for optional CUDA KEM adapter package.
    return importlib.util.find_spec("kyber_py_cuda") is not None


def cuda_sig_available() -> bool:
    # Placeholder probe for optional CUDA SIG adapter package.
    return importlib.util.find_spec("dilithium_py_cuda") is not None
