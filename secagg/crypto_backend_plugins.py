from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _env_module_name(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _prefer_liboqs() -> bool:
    return os.getenv("SECAGG_PREFER_LIBOQS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _import_first(names: Iterable[str]) -> tuple[str, Any] | None:
    for module_name in names:
        if not module_name:
            continue
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
            return module_name, importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _resolve_oqs_like_kem_name(level: str) -> list[str]:
    mapping = {
        "ML-KEM-512": ["ML-KEM-512", "Kyber512"],
        "ML-KEM-768": ["ML-KEM-768", "Kyber768"],
        "ML-KEM-1024": ["ML-KEM-1024", "Kyber1024"],
    }
    return mapping.get(level, [level])


def _resolve_oqs_like_sig_name(level: str) -> list[str]:
    mapping = {
        "ML-DSA-44": ["ML-DSA-44", "Dilithium2"],
        "ML-DSA-65": ["ML-DSA-65", "Dilithium3"],
        "ML-DSA-87": ["ML-DSA-87", "Dilithium5"],
        "SLH-DSA-shake_128f": ["SLH-DSA-shake_128f", "SPHINCS+-SHAKE-128f-simple"],
        "SLH-DSA-shake_128s": ["SLH-DSA-shake_128s", "SPHINCS+-SHAKE-128s-simple"],
        "SLH-DSA-shake_192f": ["SLH-DSA-shake_192f", "SPHINCS+-SHAKE-192f-simple"],
        "SLH-DSA-shake_192s": ["SLH-DSA-shake_192s", "SPHINCS+-SHAKE-192s-simple"],
        "SLH-DSA-shake_256f": ["SLH-DSA-shake_256f", "SPHINCS+-SHAKE-256f-simple"],
        "SLH-DSA-shake_256s": ["SLH-DSA-shake_256s", "SPHINCS+-SHAKE-256s-simple"],
        "SLH-DSA-sha2_128f": ["SLH-DSA-sha2_128f", "SPHINCS+-SHA2-128f-simple"],
        "SLH-DSA-sha2_128s": ["SLH-DSA-sha2_128s", "SPHINCS+-SHA2-128s-simple"],
        "SLH-DSA-sha2_192f": ["SLH-DSA-sha2_192f", "SPHINCS+-SHA2-192f-simple"],
        "SLH-DSA-sha2_192s": ["SLH-DSA-sha2_192s", "SPHINCS+-SHA2-192s-simple"],
        "SLH-DSA-sha2_256f": ["SLH-DSA-sha2_256f", "SPHINCS+-SHA2-256f-simple"],
        "SLH-DSA-sha2_256s": ["SLH-DSA-sha2_256s", "SPHINCS+-SHA2-256s-simple"],
    }
    return mapping.get(level, [level])


def _try_load_oqs_kem(level: str) -> tuple[str, Any] | None:
    loaded = _import_first([
        _env_module_name("SECAGG_CPU_KEM_MODULE"),
        "oqs",
        "liboqs",
    ])
    if loaded is None:
        return None
    module_name, module = loaded
    candidates = _resolve_oqs_like_kem_name(level)
    if hasattr(module, "KeyEncapsulation"):
        for candidate in candidates:
            try:
                impl = module.KeyEncapsulation(candidate)
                return module_name, OqsLikeKEMAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
    for candidate in candidates:
        for attr_name in (candidate, candidate.replace("-", "_")):
            factory = getattr(module, attr_name, None)
            if factory is None:
                continue
            try:
                impl = factory()
                return module_name, OqsLikeKEMAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
    return None


def _try_load_oqs_sig(level: str) -> tuple[str, Any] | None:
    loaded = _import_first([
        _env_module_name("SECAGG_CPU_SIG_MODULE"),
        "oqs",
        "liboqs",
    ])
    if loaded is None:
        return None
    module_name, module = loaded
    candidates = _resolve_oqs_like_sig_name(level)
    if hasattr(module, "Signature"):
        for candidate in candidates:
            try:
                impl = module.Signature(candidate)
                return module_name, OqsLikeSignatureAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
    for candidate in candidates:
        for attr_name in (candidate, candidate.replace("-", "_")):
            factory = getattr(module, attr_name, None)
            if factory is None:
                continue
            try:
                impl = factory()
                return module_name, OqsLikeSignatureAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
    return None


@dataclass
class OqsLikeKEMAdapter:
    impl: Any
    algorithm: str

    @property
    def encapsulation_key_size(self) -> int:
        for attr in ("length_public_key", "public_key_size", "encapsulation_key_size"):
            value = getattr(self.impl, attr, None)
            if isinstance(value, int):
                return value
        pk, _ = self.keygen()
        return len(pk)

    def keygen(self):
        for method_name in ("generate_keypair", "keygen"):
            method = getattr(self.impl, method_name, None)
            if callable(method):
                return method()
        raise AttributeError(f"KEM backend {self.impl!r} does not expose key generation")

    def encaps(self, public_key: bytes):
        for method_name in ("encap_secret", "encaps", "encapsulate"):
            method = getattr(self.impl, method_name, None)
            if callable(method):
                result = method(public_key)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
        raise AttributeError(f"KEM backend {self.impl!r} does not expose encapsulation")

    def decaps(self, secret_key: bytes, ciphertext: bytes):
        for method_name in ("decap_secret", "decaps", "decapsulate"):
            method = getattr(self.impl, method_name, None)
            if callable(method):
                return method(secret_key, ciphertext) if method_name in {"decaps", "decapsulate"} else method(ciphertext)
        raise AttributeError(f"KEM backend {self.impl!r} does not expose decapsulation")


@dataclass
class OqsLikeSignatureAdapter:
    impl: Any
    algorithm: str

    def keygen(self):
        for method_name in ("generate_keypair", "keygen"):
            method = getattr(self.impl, method_name, None)
            if callable(method):
                return method()
        raise AttributeError(f"Signature backend {self.impl!r} does not expose key generation")

    def sign(self, secret_key: bytes, message: bytes):
        for method_name in ("sign", "sign_message"):
            method = getattr(self.impl, method_name, None)
            if callable(method):
                try:
                    return method(message, secret_key)
                except TypeError:
                    return method(secret_key, message)
        raise AttributeError(f"Signature backend {self.impl!r} does not expose signing")

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        for method_name in ("verify", "verify_message"):
            method = getattr(self.impl, method_name, None)
            if callable(method):
                try:
                    return bool(method(message, signature, public_key))
                except TypeError:
                    try:
                        return bool(method(public_key, message, signature))
                    except TypeError:
                        return bool(method(signature, message, public_key))
        raise AttributeError(f"Signature backend {self.impl!r} does not expose verification")


def load_cuda_kem_adapter(level: str) -> tuple[str, Any] | None:
    module_names = [
        _env_module_name("SECAGG_CUDA_KEM_MODULE"),
        "liboqs_cuda",
        "pqc_cuda_kyber",
        "kyber_cuda",
    ]
    for candidate in _resolve_oqs_like_kem_name(level):
        loaded = _import_first(module_names)
        if loaded is None:
            return None
        module_name, module = loaded
        if hasattr(module, "KeyEncapsulation"):
            try:
                impl = module.KeyEncapsulation(candidate)
                return module_name, OqsLikeKEMAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
        for attr_name in (candidate, candidate.replace("-", "_")):
            factory = getattr(module, attr_name, None)
            if factory is None:
                continue
            try:
                impl = factory()
                return module_name, OqsLikeKEMAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
    return None


def load_cuda_sig_adapter(level: str) -> tuple[str, Any] | None:
    module_names = [
        _env_module_name("SECAGG_CUDA_SIG_MODULE"),
        "cuDilithium",
        "dilithium_cuda",
        "liboqs_cuda",
    ]
    for candidate in _resolve_oqs_like_sig_name(level):
        loaded = _import_first(module_names)
        if loaded is None:
            return None
        module_name, module = loaded
        if hasattr(module, "Signature"):
            try:
                impl = module.Signature(candidate)
                return module_name, OqsLikeSignatureAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
        for attr_name in (candidate, candidate.replace("-", "_")):
            factory = getattr(module, attr_name, None)
            if factory is None:
                continue
            try:
                impl = factory()
                return module_name, OqsLikeSignatureAdapter(impl=impl, algorithm=candidate)
            except Exception:
                continue
    return None


def load_cpu_kem_adapter(level: str) -> tuple[str, Any] | None:
    if not _prefer_liboqs():
        return None
    return _try_load_oqs_kem(level)


def load_cpu_sig_adapter(level: str) -> tuple[str, Any] | None:
    if not _prefer_liboqs():
        return None
    return _try_load_oqs_sig(level)
