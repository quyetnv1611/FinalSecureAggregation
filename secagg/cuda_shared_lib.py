from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path


_UINT8_PTR = ctypes.c_void_p
_SIZE_T_PTR = ctypes.POINTER(ctypes.c_size_t)


def _load_library(path: str | None, fallbacks: tuple[str, ...]) -> tuple[str, ctypes.CDLL] | None:
    if path:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"CUDA shared library not found: {expanded}")
        return expanded, ctypes.CDLL(expanded)

    search_roots = []
    for root in (
        os.getenv("SECAGG_CUDA_LIBRARY_ROOT", ""),
        "/content/drive",
        "/content",
        os.getcwd(),
    ):
        if root and root not in search_roots:
            search_roots.append(root)

    for root in search_roots:
        base = Path(os.path.expandvars(os.path.expanduser(root)))
        if not base.exists():
            continue
        for candidate in fallbacks:
            for match in base.rglob(candidate):
                if match.is_file():
                    return str(match), ctypes.CDLL(str(match))

    for candidate in fallbacks:
        try:
            return candidate, ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def _buffer_ptr(buffer: ctypes.Array) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(buffer))


def _bytes_to_buffer(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


@dataclass
class CTypesMLKEMAdapter:
    lib: ctypes.CDLL
    algorithm: str

    _SIZE_MAP = {
        "ML-KEM-512": (800, 1632, 768, 32),
        "ML-KEM-768": (1184, 2400, 1088, 32),
        "ML-KEM-1024": (1568, 3168, 1568, 32),
    }

    @classmethod
    def from_library(cls, lib: ctypes.CDLL, algorithm: str) -> "CTypesMLKEMAdapter":
        return cls(lib=lib, algorithm=algorithm)

    @property
    def length_public_key(self) -> int:
        return self._SIZE_MAP[self.algorithm][0]

    @property
    def length_secret_key(self) -> int:
        return self._SIZE_MAP[self.algorithm][1]

    @property
    def length_ciphertext(self) -> int:
        return self._SIZE_MAP[self.algorithm][2]

    @property
    def length_shared_secret(self) -> int:
        return self._SIZE_MAP[self.algorithm][3]

    def keygen(self):
        public_key = ctypes.create_string_buffer(self.length_public_key)
        secret_key = ctypes.create_string_buffer(self.length_secret_key)
        func = getattr(self.lib, f"OQS_KEM_{self.algorithm.lower().replace('-', '_')}_keypair")
        func.argtypes = [_UINT8_PTR, _UINT8_PTR]
        func.restype = ctypes.c_int
        result = func(_buffer_ptr(public_key), _buffer_ptr(secret_key))
        if result != 0:
            raise RuntimeError(f"KEM keypair failed for {self.algorithm} (rc={result})")
        return public_key.raw[: self.length_public_key], secret_key.raw[: self.length_secret_key]

    def encaps(self, public_key: bytes):
        if len(public_key) != self.length_public_key:
            raise ValueError(
                f"Expected {self.length_public_key} public key bytes for {self.algorithm}, got {len(public_key)}"
            )
        ciphertext = ctypes.create_string_buffer(self.length_ciphertext)
        shared_secret = ctypes.create_string_buffer(self.length_shared_secret)
        func = getattr(self.lib, f"OQS_KEM_{self.algorithm.lower().replace('-', '_')}_encaps")
        func.argtypes = [_UINT8_PTR, _UINT8_PTR, _UINT8_PTR]
        func.restype = ctypes.c_int
        result = func(_buffer_ptr(ciphertext), _buffer_ptr(shared_secret), _buffer_ptr(_bytes_to_buffer(public_key)))
        if result != 0:
            raise RuntimeError(f"KEM encaps failed for {self.algorithm} (rc={result})")
        return shared_secret.raw[: self.length_shared_secret], ciphertext.raw[: self.length_ciphertext]

    def decaps(self, secret_key: bytes, ciphertext: bytes):
        if len(secret_key) != self.length_secret_key:
            raise ValueError(
                f"Expected {self.length_secret_key} secret key bytes for {self.algorithm}, got {len(secret_key)}"
            )
        if len(ciphertext) != self.length_ciphertext:
            raise ValueError(
                f"Expected {self.length_ciphertext} ciphertext bytes for {self.algorithm}, got {len(ciphertext)}"
            )
        shared_secret = ctypes.create_string_buffer(self.length_shared_secret)
        func = getattr(self.lib, f"OQS_KEM_{self.algorithm.lower().replace('-', '_')}_decaps")
        func.argtypes = [_UINT8_PTR, _UINT8_PTR, _UINT8_PTR]
        func.restype = ctypes.c_int
        result = func(
            _buffer_ptr(shared_secret),
            _buffer_ptr(_bytes_to_buffer(ciphertext)),
            _buffer_ptr(_bytes_to_buffer(secret_key)),
        )
        if result != 0:
            raise RuntimeError(f"KEM decaps failed for {self.algorithm} (rc={result})")
        return shared_secret.raw[: self.length_shared_secret]


@dataclass
class CTypesDilithiumAdapter:
    lib: ctypes.CDLL
    algorithm: str

    _SIZE_MAP = {
        "ML-DSA-44": (1312, 2560, 2420),
        "ML-DSA-65": (1952, 4032, 3309),
        "ML-DSA-87": (2592, 4896, 4627),
    }

    @classmethod
    def from_library(cls, lib: ctypes.CDLL, algorithm: str) -> "CTypesDilithiumAdapter":
        return cls(lib=lib, algorithm=algorithm)

    @property
    def length_public_key(self) -> int:
        return self._SIZE_MAP[self.algorithm][0]

    @property
    def length_secret_key(self) -> int:
        return self._SIZE_MAP[self.algorithm][1]

    @property
    def length_signature(self) -> int:
        return self._SIZE_MAP[self.algorithm][2]

    def keygen(self):
        public_key = ctypes.create_string_buffer(self.length_public_key)
        secret_key = ctypes.create_string_buffer(self.length_secret_key)
        func = getattr(self.lib, "crypto_sign_keypair")
        func.argtypes = [_UINT8_PTR, _UINT8_PTR]
        func.restype = ctypes.c_int
        result = func(_buffer_ptr(public_key), _buffer_ptr(secret_key))
        if result != 0:
            raise RuntimeError(f"Signature keypair failed for {self.algorithm} (rc={result})")
        return public_key.raw[: self.length_public_key], secret_key.raw[: self.length_secret_key]

    def sign(self, secret_key: bytes, message: bytes):
        if len(secret_key) != self.length_secret_key:
            raise ValueError(
                f"Expected {self.length_secret_key} secret key bytes for {self.algorithm}, got {len(secret_key)}"
            )
        signature = ctypes.create_string_buffer(self.length_signature)
        siglen = ctypes.c_size_t(self.length_signature)
        ctx = ctypes.create_string_buffer(1)
        func = getattr(self.lib, "crypto_sign_signature")
        func.argtypes = [_UINT8_PTR, _SIZE_T_PTR, _UINT8_PTR, ctypes.c_size_t, _UINT8_PTR, ctypes.c_size_t, _UINT8_PTR]
        func.restype = ctypes.c_int
        result = func(
            _buffer_ptr(signature),
            ctypes.byref(siglen),
            _buffer_ptr(_bytes_to_buffer(message)),
            len(message),
            _buffer_ptr(ctx),
            0,
            _buffer_ptr(_bytes_to_buffer(secret_key)),
        )
        if result != 0:
            raise RuntimeError(f"Signature sign failed for {self.algorithm} (rc={result})")
        return signature.raw[: siglen.value]

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        if len(public_key) != self.length_public_key:
            raise ValueError(
                f"Expected {self.length_public_key} public key bytes for {self.algorithm}, got {len(public_key)}"
            )
        func = getattr(self.lib, "crypto_sign_verify")
        func.argtypes = [_UINT8_PTR, ctypes.c_size_t, _UINT8_PTR, ctypes.c_size_t, _UINT8_PTR]
        func.restype = ctypes.c_int
        result = func(
            _buffer_ptr(_bytes_to_buffer(signature)),
            len(signature),
            _buffer_ptr(_bytes_to_buffer(message)),
            len(message),
            _buffer_ptr(_bytes_to_buffer(public_key)),
        )
        return result == 0


def load_ctypes_mlkem_adapter(algorithm: str, library_path: str | None = None) -> tuple[str, CTypesMLKEMAdapter] | None:
    loaded = _load_library(
        library_path,
        (
            "liboqs.so",
            "oqs.dll",
            "liboqs.dylib",
        ),
    )
    if loaded is None:
        return None
    lib_name, lib = loaded
    return lib_name, CTypesMLKEMAdapter.from_library(lib, algorithm)


def load_ctypes_dilithium_adapter(algorithm: str, library_path: str | None = None) -> tuple[str, CTypesDilithiumAdapter] | None:
    loaded = _load_library(
        library_path,
        (
            "libcuDilithium3.so",
            "libcuDilithium2.so",
            "libcuDilithium5.so",
            "cuDilithium3.dll",
            "cuDilithium2.dll",
            "cuDilithium5.dll",
            "libcuDilithium3.dylib",
            "libcuDilithium2.dylib",
            "libcuDilithium5.dylib",
        ),
    )
    if loaded is None:
        return None
    lib_name, lib = loaded
    return lib_name, CTypesDilithiumAdapter.from_library(lib, algorithm)