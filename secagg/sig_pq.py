from __future__ import annotations

import abc
import hashlib
import logging
import os
from typing import Tuple

from .crypto_backend import cuda_sig_available, resolve_mode, split_backend_qualifier
from .crypto_backend_plugins import load_cpu_sig_adapter, load_cuda_sig_adapter

MSG = b"Secure Aggregation -- sign this public key broadcast"

import time

logger = logging.getLogger(__name__)
PublicKeyBytes = bytes
SecretKeyBytes = bytes
SignatureBytes = bytes

_CUDA_SIG_WARNED = False

# --- ClassicECDSASigner ---
class ClassicECDSASigner:
    name = "classic-ECDSA-P256"
    is_post_quantum = False

    def __init__(self) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key, ECDH
            from cryptography.hazmat.backends import default_backend
            self._backend = default_backend()
            self._curve = SECP256R1()
            self._use_real = True
            logger.debug("ClassicECDSASigner: using ECDSA-P256 via cryptography.")
        except ImportError:
            self._use_real = False
            logger.warning("cryptography package not installed. ClassicECDSASigner will use HMAC-SHA256 (NOT a real signature — install cryptography: pip install cryptography).")

    def keygen(self) -> Tuple[PublicKeyBytes, SecretKeyBytes]:
        if self._use_real:
            from cryptography.hazmat.primitives.asymmetric.ec import generate_private_key, SECP256R1
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
            sk_obj = generate_private_key(SECP256R1(), self._backend)
            pk_bytes = sk_obj.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
            sk_bytes = sk_obj.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
            return pk_bytes, sk_bytes
        else:
            sk = os.urandom(32)
            pk = hashlib.sha256(b"hmac_pk" + sk).digest()
            return pk, sk

    def sign(self, sk: SecretKeyBytes, message: bytes) -> SignatureBytes:
        if self._use_real:
            from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
            from cryptography.hazmat.primitives.hashes import SHA256
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            sk_obj = load_pem_private_key(sk, password=None, backend=self._backend)
            return sk_obj.sign(message, ECDSA(SHA256()))
        else:
            import hmac
            return hmac.new(sk, message, hashlib.sha256).digest()

    def verify(self, pk: PublicKeyBytes, message: bytes, signature: SignatureBytes) -> bool:
        if self._use_real:
            try:
                from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
                from cryptography.hazmat.primitives.hashes import SHA256
                pk_obj = self._load_ecdsa_pk(pk, self._backend)
                pk_obj.verify(signature, message, ECDSA(SHA256()))
                return True
            except Exception:
                return False
        else:
            return False


    @staticmethod
    def _load_ecdsa_pk(pk_bytes: bytes, backend):
        from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicNumbers, SECP256R1
        assert pk_bytes[0] == 0x04 and len(pk_bytes) == 65
        x = int.from_bytes(pk_bytes[1:33], "big")
        y = int.from_bytes(pk_bytes[33:65], "big")
        return EllipticCurvePublicNumbers(x, y, SECP256R1()).public_key(backend)
class DilithiumSigner:
    is_post_quantum = True
    _VARIANTS = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")

    def __init__(self, level: str = "ML-DSA-65") -> None:
        if level not in self._VARIANTS:
            raise ValueError(f"Unknown Dilithium level '{level}'. Choose from: {self._VARIANTS}")
        self._level = level
        try:
            from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
            self._impl = {
                "ML-DSA-44": ML_DSA_44,
                "ML-DSA-65": ML_DSA_65,
                "ML-DSA-87": ML_DSA_87,
            }[level]
            logger.debug("DilithiumSigner: %s loaded via dilithium-py.", level)
        except ImportError as exc:
            raise ImportError("dilithium-py not installed.  Run:  pip install dilithium-py") from exc

    @property
    def name(self) -> str:
        return self._level

    def keygen(self) -> Tuple[PublicKeyBytes, SecretKeyBytes]:
        pk, sk = self._impl.keygen()
        return pk, sk

    def sign(self, sk: SecretKeyBytes, message: bytes) -> SignatureBytes:
        return self._impl.sign(sk, message)

    def verify(self, pk: PublicKeyBytes, message: bytes, signature: SignatureBytes) -> bool:
        try:
            return bool(self._impl.verify(pk, message, signature))
        except Exception:
            return False


# --- SPHINCSPlusSigner ---
_SLHDSA_PARAM_MAP: dict = {}
def _init_slhdsa_map() -> None:
    try:
        import slhdsa as _slh
        _SLHDSA_PARAM_MAP.update({
            "SLH-DSA-shake_128s":  _slh.shake_128s,
            "SLH-DSA-shake_128f":  _slh.shake_128f,
            "SLH-DSA-shake_192s":  _slh.shake_192s,
            "SLH-DSA-shake_192f":  _slh.shake_192f,
            "SLH-DSA-shake_256s":  _slh.shake_256s,
            "SLH-DSA-shake_256f":  _slh.shake_256f,
            "SLH-DSA-sha2_128s":   _slh.sha2_128s,
            "SLH-DSA-sha2_128f":   _slh.sha2_128f,
            "SLH-DSA-sha2_192s":   _slh.sha2_192s,
            "SLH-DSA-sha2_192f":   _slh.sha2_192f,
            "SLH-DSA-sha2_256s":   _slh.sha2_256s,
            "SLH-DSA-sha2_256f":   _slh.sha2_256f,
        })
    except ImportError:
        pass
_init_slhdsa_map()

class SPHINCSPlusSigner:
    is_post_quantum = True

    def __init__(self, variant: str = "SLH-DSA-shake_128f") -> None:
        if variant not in _SLHDSA_PARAM_MAP:
            if not _SLHDSA_PARAM_MAP:
                raise ImportError("slh-dsa not installed.  Run:  pip install slh-dsa")
            raise ValueError(f"Unknown SLH-DSA variant '{variant}'. Available: {sorted(_SLHDSA_PARAM_MAP)}")
        self._variant = variant
        self._par = _SLHDSA_PARAM_MAP[variant]
        logger.debug("SPHINCSPlusSigner: %s loaded via slhdsa.", variant)

    @property
    def name(self) -> str:
        return self._variant

    def keygen(self) -> Tuple[PublicKeyBytes, SecretKeyBytes]:
        import slhdsa.lowlevel.slhdsa as _ll
        from slhdsa.slhdsa import SecretKey as _SK
        sk_tuple, _pk_tuple = _ll.keygen(self._par)
        sk_obj = _SK(sk_tuple, self._par)
        pk_obj = sk_obj.pubkey
        return pk_obj.digest(), sk_obj.digest()

    def sign(self, sk: SecretKeyBytes, message: bytes) -> SignatureBytes:
        from slhdsa.slhdsa import SecretKey as _SK
        sk_obj = _SK.from_digest(sk, self._par)
        return sk_obj.sign_pure(message)

    def verify(self, pk: PublicKeyBytes, message: bytes, signature: SignatureBytes) -> bool:
        from slhdsa.slhdsa import PublicKey as _PK
        from slhdsa.exception import SLHDSAVerifyException
        try:
            pk_obj = _PK.from_digest(pk, self._par)
            return bool(pk_obj.verify_pure(message, signature))
        except (SLHDSAVerifyException, Exception):
            return False

# --- Factory ---




def make_signer(backend: str):
    global _CUDA_SIG_WARNED

    base_backend, explicit_mode = split_backend_qualifier(backend)
    accel_mode = resolve_mode(explicit_mode)
    if accel_mode == "cuda":
        cuda_loaded = load_cuda_sig_adapter(base_backend)
        if cuda_loaded is not None:
            module_name, adapter = cuda_loaded
            logger.info("Using CUDA signature adapter %s for %s.", module_name, base_backend)
            return adapter
        if not cuda_sig_available() and not _CUDA_SIG_WARNED:
            logger.warning(
                "CUDA mode requested for signatures but no CUDA signature adapter was found; "
                "falling back to CPU backend."
            )
            _CUDA_SIG_WARNED = True

    cpu_loaded = load_cpu_sig_adapter(base_backend)
    if cpu_loaded is not None:
        module_name, adapter = cpu_loaded
        logger.info("Using liboqs CPU signature adapter %s for %s.", module_name, base_backend)
        return adapter

    if base_backend in ("classic", "ECDSA-P256", "ecdsa"):
        return ClassicECDSASigner()
    if base_backend in ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87"):
        return DilithiumSigner(level=base_backend)
    if base_backend.startswith("SLH-DSA-"):
        return SPHINCSPlusSigner(variant=base_backend)
    raise ValueError(
        f"Unknown signature backend '{base_backend}'.\n"
        "Valid options:\n"
        "  Classic : 'classic' / 'ECDSA-P256'\n"
        "  FIPS 204: 'ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87'\n"
        "  FIPS 205: 'SLH-DSA-shake_128f', 'SLH-DSA-shake_128s',\n"
        "            'SLH-DSA-sha2_256s', 'SLH-DSA-sha2_256f', … (12 variants)"
    )

# --- Main block for benchmarking ---
if __name__ == "__main__":
    import time
    configs = [
        ("classic",              "Classical ECDSA-P256      "),
        ("ML-DSA-44",            "Dilithium ML-DSA-44        "),
        ("ML-DSA-65",            "Dilithium ML-DSA-65        "),
        ("ML-DSA-87",            "Dilithium ML-DSA-87        "),
        ("SLH-DSA-shake_128f",   "SPHINCS+ SLH-DSA-shake_128f"),
        ("SLH-DSA-shake_128s",   "SPHINCS+ SLH-DSA-shake_128s"),
        ("SLH-DSA-sha2_256s",    "SPHINCS+ SLH-DSA-sha2_256s "),
    ]
    print(f"\n{'Backend':<32} {'pk(B)':>7} {'sk(B)':>7} {'sig(B)':>8} "
          f"{'keygen(ms)':>12} {'sign(ms)':>10} {'verify(ms)':>11} {'ok':>4}  PQ?")
    print("-" * 105)
    for backend_id, label in configs:
        try:
            s = make_signer(backend_id)
            t0 = time.perf_counter()
            pk, sk = s.keygen()
            t_kg = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            sig = s.sign(sk, MSG)
            t_sign = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            ok = s.verify(pk, MSG, sig)
            t_ver = (time.perf_counter() - t0) * 1000
            pq = "✅" if s.is_post_quantum else "❌"
            print(f"{label:<32} {len(pk):>7} {len(sk):>7} {len(sig):>8} "
                  f"{t_kg:>11.1f}  {t_sign:>9.1f}  {t_ver:>10.1f}  "
                  f"{'✅' if ok else '❌'}  {pq}")
        except Exception as exc:
            print(f"{label:<32}  ERROR: {exc}")
    print()
    _VARIANTS = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")

    def __init__(self, level: str = "ML-DSA-65") -> None:
        if level not in self._VARIANTS:
            raise ValueError(
                f"Unknown Dilithium level '{level}'. "
                f"Choose from: {self._VARIANTS}"
            )
        self._level = level
        try:
            from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
            self._impl = {
                "ML-DSA-44": ML_DSA_44,
                "ML-DSA-65": ML_DSA_65,
                "ML-DSA-87": ML_DSA_87,
            }[level]
            logger.debug("DilithiumSigner: %s loaded via dilithium-py.", level)
        except ImportError as exc:
            raise ImportError(
                "dilithium-py not installed.  Run:  pip install dilithium-py"
            ) from exc

    @property
    def name(self) -> str:
        return self._level

    def keygen(self) -> Tuple[PublicKeyBytes, SecretKeyBytes]:
        # Generate an ML-DSA key pair
        pk, sk = self._impl.keygen()
        return pk, sk

    def sign(self, sk: SecretKeyBytes, message: bytes) -> SignatureBytes:
        # Sign *message* using ML-DSA (deterministic by default in FIPS 204)
        return self._impl.sign(sk, message)

    def verify(
        self,
        pk: PublicKeyBytes,
        message: bytes,
        signature: SignatureBytes,
    ) -> bool:
        # Verify an ML-DSA signature
        try:
            return bool(self._impl.verify(pk, message, signature))
        except Exception:
            return False




# Map friendly names → slhdsa parameter objects (populated at import time)
_SLHDSA_PARAM_MAP: dict = {}

def _init_slhdsa_map() -> None:
    # Populate _SLHDSA_PARAM_MAP once the library is imported
    try:
        import slhdsa as _slh
        _SLHDSA_PARAM_MAP.update({
            # SHAKE variants
            "SLH-DSA-shake_128s":  _slh.shake_128s,
            "SLH-DSA-shake_128f":  _slh.shake_128f,
            "SLH-DSA-shake_192s":  _slh.shake_192s,
            "SLH-DSA-shake_192f":  _slh.shake_192f,
            "SLH-DSA-shake_256s":  _slh.shake_256s,
            "SLH-DSA-shake_256f":  _slh.shake_256f,
            # SHA-2 variants
            "SLH-DSA-sha2_128s":   _slh.sha2_128s,
            "SLH-DSA-sha2_128f":   _slh.sha2_128f,
            "SLH-DSA-sha2_192s":   _slh.sha2_192s,
            "SLH-DSA-sha2_192f":   _slh.sha2_192f,
            "SLH-DSA-sha2_256s":   _slh.sha2_256s,
            "SLH-DSA-sha2_256f":   _slh.sha2_256f,
        })
    except ImportError:
        pass  # handled lazily in SPHINCSPlusSigner.__init__



_init_slhdsa_map()

class SPHINCSPlusSigner:
    is_post_quantum = True

    def __init__(self, variant: str = "SLH-DSA-shake_128f") -> None:
        if variant not in _SLHDSA_PARAM_MAP:
            if not _SLHDSA_PARAM_MAP:
                raise ImportError(
                    "slh-dsa not installed.  Run:  pip install slh-dsa"
                )
            raise ValueError(
                f"Unknown SLH-DSA variant '{variant}'. "
                f"Available: {sorted(_SLHDSA_PARAM_MAP)}"
            )
        self._variant = variant
        self._par = _SLHDSA_PARAM_MAP[variant]
        logger.debug("SPHINCSPlusSigner: %s loaded via slhdsa.", variant)

    @property
    def name(self) -> str:
        return self._variant

    def keygen(self) -> Tuple[PublicKeyBytes, SecretKeyBytes]:
        # Generate a SLH-DSA key pair
        import slhdsa.lowlevel.slhdsa as _ll
        from slhdsa.slhdsa import SecretKey as _SK
        sk_tuple, _pk_tuple = _ll.keygen(self._par)
        # sk_tuple = (SK_seed, SK_prf, PK_seed, PK_root) — 4-part tuple
        sk_obj = _SK(sk_tuple, self._par)
        pk_obj = sk_obj.pubkey                      # derived from sk
        return pk_obj.digest(), sk_obj.digest()     # (pk_bytes, sk_bytes)

    def sign(self, sk: SecretKeyBytes, message: bytes) -> SignatureBytes:
        # Sign *message* using SLH-DSA Pure API (FIPS 205 §9.2)
        from slhdsa.slhdsa import SecretKey as _SK
        sk_obj = _SK.from_digest(sk, self._par)
        return sk_obj.sign_pure(message)

    def verify(
        self,
        pk: PublicKeyBytes,
        message: bytes,
        signature: SignatureBytes,
    ) -> bool:
        # Verify a SLH-DSA Pure signature (FIPS 205 §9.3)
        from slhdsa.slhdsa import PublicKey as _PK
        from slhdsa.exception import SLHDSAVerifyException
        try:
            pk_obj = _PK.from_digest(pk, self._par)
            return bool(pk_obj.verify_pure(message, signature))
        except (SLHDSAVerifyException, Exception):
            return False







    configs = [
        ("classic",              "Classical ECDSA-P256      "),
        ("ML-DSA-44",            "Dilithium ML-DSA-44        "),
        ("ML-DSA-65",            "Dilithium ML-DSA-65        "),
        ("ML-DSA-87",            "Dilithium ML-DSA-87        "),
        ("SLH-DSA-shake_128f",   "SPHINCS+ SLH-DSA-shake_128f"),
        ("SLH-DSA-shake_128s",   "SPHINCS+ SLH-DSA-shake_128s"),
        ("SLH-DSA-sha2_256s",    "SPHINCS+ SLH-DSA-sha2_256s "),
    ]

    print(f"\n{'Backend':<32} {'pk(B)':>7} {'sk(B)':>7} {'sig(B)':>8} "
          f"{'keygen(ms)':>12} {'sign(ms)':>10} {'verify(ms)':>11} {'ok':>4}  PQ?")
    print("-" * 105)

    for backend_id, label in configs:
        try:
            s = make_signer(backend_id)

            t0 = time.perf_counter()
            pk, sk = s.keygen()
            t_kg = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            sig = s.sign(sk, MSG)
            t_sign = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            ok = s.verify(pk, MSG, sig)
            t_ver = (time.perf_counter() - t0) * 1000

            pq = "✅" if s.is_post_quantum else "❌"
            print(f"{label:<32} {len(pk):>7} {len(sk):>7} {len(sig):>8} "
                  f"{t_kg:>11.1f}  {t_sign:>9.1f}  {t_ver:>10.1f}  "
                  f"{'✅' if ok else '❌'}  {pq}")
        except Exception as exc:
            print(f"{label:<32}  ERROR: {exc}")

    print()
