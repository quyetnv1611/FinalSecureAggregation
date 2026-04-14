from __future__ import annotations

"""Skeleton adapter for optional CUDA crypto backends.

This module is intentionally a thin compatibility layer so you can plug an
external CUDA implementation without changing benchmark code. It currently
falls back to existing CPU libraries while preserving adapter API shape.

Usage example:
  python experiments/benchmarks/bench_orig_vs_pq.py \
      --crypto-accel cuda \
      --cuda-kem-module secagg.cuda_adapter_skeleton \
      --cuda-sig-module secagg.cuda_adapter_skeleton
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Tuple


def _warn_once() -> None:
    # Keep this explicit so users don't mistake this skeleton for true CUDA.
    if os.getenv("SECAGG_CUDA_SKELETON_WARNED") == "1":
        return
    os.environ["SECAGG_CUDA_SKELETON_WARNED"] = "1"
    print("[secagg] cuda_adapter_skeleton loaded: using CPU fallback implementation.")


@dataclass
class _KEMShim:
    impl: object
    public_key_size: int

    def keygen(self):
        return self.impl.keygen()

    def encaps(self, public_key: bytes):
        # Existing repo expects (shared_secret, ciphertext)
        return self.impl.encaps(public_key)

    def decaps(self, secret_key: bytes, ciphertext: bytes):
        return self.impl.decaps(secret_key, ciphertext)

    @property
    def encapsulation_key_size(self) -> int:
        return self.public_key_size


def ML_KEM_512() -> _KEMShim:
    _warn_once()
    from kyber_py.ml_kem import ML_KEM_512  # type: ignore

    pk, _ = ML_KEM_512.keygen()
    return _KEMShim(impl=ML_KEM_512, public_key_size=len(pk))


def ML_KEM_768() -> _KEMShim:
    _warn_once()
    from kyber_py.ml_kem import ML_KEM_768  # type: ignore

    pk, _ = ML_KEM_768.keygen()
    return _KEMShim(impl=ML_KEM_768, public_key_size=len(pk))


def ML_KEM_1024() -> _KEMShim:
    _warn_once()
    from kyber_py.ml_kem import ML_KEM_1024  # type: ignore

    pk, _ = ML_KEM_1024.keygen()
    return _KEMShim(impl=ML_KEM_1024, public_key_size=len(pk))


class _DilithiumShim:
    def __init__(self, level: str) -> None:
        _warn_once()
        from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87  # type: ignore

        self._impl = {
            "ML-DSA-44": ML_DSA_44,
            "ML-DSA-65": ML_DSA_65,
            "ML-DSA-87": ML_DSA_87,
        }[level]

    def keygen(self):
        return self._impl.keygen()

    def sign(self, secret_key: bytes, message: bytes):
        return self._impl.sign(secret_key, message)

    def verify(self, public_key: bytes, message: bytes, signature: bytes):
        return bool(self._impl.verify(public_key, message, signature))


def ML_DSA_44():
    return _DilithiumShim("ML-DSA-44")


def ML_DSA_65():
    return _DilithiumShim("ML-DSA-65")


def ML_DSA_87():
    return _DilithiumShim("ML-DSA-87")


class _SPHINCSShim:
    def __init__(self, variant: str) -> None:
        _warn_once()
        import slhdsa  # type: ignore
        import slhdsa.lowlevel.slhdsa as _ll  # type: ignore
        from slhdsa.slhdsa import PublicKey, SecretKey  # type: ignore

        mapping = {
            "SLH-DSA-shake_128f": slhdsa.shake_128f,
            "SLH-DSA-shake_128s": slhdsa.shake_128s,
            "SLH-DSA-sha2_256f": slhdsa.sha2_256f,
            "SLH-DSA-sha2_256s": slhdsa.sha2_256s,
        }
        self._par = mapping[variant]
        self._ll = _ll
        self._PublicKey = PublicKey
        self._SecretKey = SecretKey

    def keygen(self):
        sk_tuple, _ = self._ll.keygen(self._par)
        sk_obj = self._SecretKey(sk_tuple, self._par)
        return sk_obj.pubkey.digest(), sk_obj.digest()

    def sign(self, secret_key: bytes, message: bytes):
        sk_obj = self._SecretKey.from_digest(secret_key, self._par)
        return sk_obj.sign_pure(message)

    def verify(self, public_key: bytes, message: bytes, signature: bytes):
        pk_obj = self._PublicKey.from_digest(public_key, self._par)
        return bool(pk_obj.verify_pure(message, signature))


def SLH_DSA_shake_128f():
    return _SPHINCSShim("SLH-DSA-shake_128f")


def SLH_DSA_shake_128s():
    return _SPHINCSShim("SLH-DSA-shake_128s")


def SLH_DSA_sha2_256f():
    return _SPHINCSShim("SLH-DSA-sha2_256f")


def SLH_DSA_sha2_256s():
    return _SPHINCSShim("SLH-DSA-sha2_256s")
