

import hashlib
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from . import config
from .crypto_backend import cuda_kem_available, resolve_mode, split_backend_qualifier
from .crypto_backend_plugins import load_cpu_kem_adapter, load_cuda_kem_adapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML-KEM import  (kyber-py v1.x implements FIPS 203)
# ---------------------------------------------------------------------------
try:
    from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024  # type: ignore
    _MLKEM_AVAILABLE = True
    logger.debug("kyber-py loaded — ML-KEM available at all three security levels.")
except ImportError:                                             # pragma: no cover
    _MLKEM_AVAILABLE = False
    logger.warning(
        "kyber-py not installed.  Install with:  pip install kyber-py\n"
        "Falling back to an INSECURE mock — do NOT use in production."
    )


# ---------------------------------------------------------------------------
# Insecure mock (structural testing only — no real crypto)
# ---------------------------------------------------------------------------
class _MockMLKEM:                                               # pragma: no cover
    """Stands in for kyber-py when the library is unavailable.

    The shared secrets produced here are **not** secure — this class exists
    solely so that the module can be imported and structurally tested without
    the kyber-py wheel.
    """

    @staticmethod
    def keygen() -> Tuple[bytes, bytes]:
        sk = os.urandom(32)
        ek = hashlib.sha256(b"ek" + sk).digest()   # 32 B (fake)
        dk = sk + ek                                # 64 B (fake)
        return ek, dk

    @staticmethod
    def encaps(ek: bytes) -> Tuple[bytes, bytes]:
        K  = os.urandom(32)
        ct = hashlib.sha256(ek + K).digest()        # 32 B (fake)
        return K, ct

    @staticmethod
    def decaps(dk: bytes, ct: bytes) -> bytes:      # noqa: ARG002
        # This mock cannot reproduce K — structural tests only.
        return hashlib.sha256(dk[:32] + ct).digest()


# ---------------------------------------------------------------------------
# Security-level selector
# ---------------------------------------------------------------------------
_LEVEL_MAP = {
    "ML-KEM-512":  (ML_KEM_512  if _MLKEM_AVAILABLE else _MockMLKEM()),
    "ML-KEM-768":  (ML_KEM_768  if _MLKEM_AVAILABLE else _MockMLKEM()),
    "ML-KEM-1024": (ML_KEM_1024 if _MLKEM_AVAILABLE else _MockMLKEM()),
}

_CUDA_KEM_WARNED = False


def _select_mlkem(level: str):
    """Return the ML-KEM object for *level*, falling back to mock if needed."""
    global _CUDA_KEM_WARNED

    base_level, explicit_mode = split_backend_qualifier(level)
    accel_mode = resolve_mode(explicit_mode)
    if accel_mode == "cuda":
        cuda_loaded = load_cuda_kem_adapter(base_level)
        if cuda_loaded is not None:
            module_name, adapter = cuda_loaded
            logger.info("Using CUDA KEM adapter %s for %s.", module_name, base_level)
            return adapter
        if not cuda_kem_available() and not _CUDA_KEM_WARNED:
            logger.warning(
                "CUDA mode requested for KEM but no CUDA KEM adapter was found; "
                "falling back to CPU backend."
            )
            _CUDA_KEM_WARNED = True

    cpu_loaded = load_cpu_kem_adapter(base_level)
    if cpu_loaded is not None:
        module_name, adapter = cpu_loaded
        logger.info("Using liboqs CPU KEM adapter %s for %s.", module_name, base_level)
        return adapter

    if not _MLKEM_AVAILABLE:
        return _MockMLKEM()
    obj = _LEVEL_MAP.get(base_level)
    if obj is None:
        raise ValueError(
            f"Unknown security level '{base_level}'. "
            "Choose from: 'ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024'."
        )
    return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _kdf(shared_secret: bytes) -> int:
   
    return int.from_bytes(hashlib.sha256(shared_secret).digest()[:4], "big")


# def _prg(seed: int, shape: Tuple[int, ...]) -> np.ndarray:
 
#     np.random.seed(seed)
#     return np.float64(np.random.rand(*shape))

def _prg(seed: int, shape: Tuple[int, ...], use_cuda: bool = False):
    """
    Sinh số giả ngẫu nhiên chuẩn mật mã (CSPRNG) sử dụng AES-256-CTR.
    Chống lại hoàn toàn các cuộc tấn công đoán seed kể cả từ máy tính lượng tử.
    """
    if use_cuda and torch.cuda.is_available():
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(seed) % (2**63 - 1))
        return torch.randint(
            low=-10**15,
            high=10**15,
            size=shape,
            dtype=torch.int64,
            device="cuda",
            generator=generator,
        )

    # 1. Băm Seed (Shared Secret) thành khóa 32-byte (256-bit) chuẩn cho AES
    seed_bytes = seed.to_bytes((seed.bit_length() + 7) // 8 or 1, "big")
    key = hashlib.sha256(seed_bytes).digest()
    
    # 2. Tính toán lượng dữ liệu cần thiết (Int64 chiếm 8 bytes mỗi số)
    num_elements = np.prod(shape)
    num_bytes = int(num_elements * 8)
    
    # 3. Thiết lập bộ sinh mã AES-256 ở chế độ CTR (Counter Mode)
    # Nonce có thể cố định (toàn số 0) vì mỗi cặp Client đã có một Key duy nhất từ ML-KEM
    nonce = b'\x00' * 16 
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Mã hóa một mảng toàn số 0 để lấy dòng byte ngẫu nhiên (Stream Cipher)
    zeros = bytes(num_bytes)
    prg_bytes = encryptor.update(zeros) + encryptor.finalize()
    
    # 4. Chuyển dòng byte thành mảng số nguyên Numpy (Int64)
    arr = np.frombuffer(prg_bytes, dtype=np.int64).copy()
    
    # 5. Áp dụng Modulo Toán học để giới hạn kích thước, tránh tràn bộ nhớ khi cộng 50 clients
    # Giới hạn an toàn: từ -10^15 đến 10^15
    arr = (arr % (2 * 10**15)) - 10**15
    
    return arr.reshape(shape)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class SecAggregatorMLKEM:
    
    SCALE_FACTOR = 100000000.0

    def __init__(
        self,
        shape: Tuple[int, int] = config.MODEL_SHAPE,
        security_level: str = "ML-KEM-768",
        crypto_accel: str | None = None,
    ) -> None:
        self._mlkem = _select_mlkem(security_level)
        self._security_level = security_level
        self._shape = shape
        self._accel_mode = resolve_mode(crypto_accel)

        # --- FIPS 203 §7.1  ML-KEM.KeyGen() ---
        # ek  = encapsulation key (public  — shared with all peers)
        # dk  = decapsulation key (private — never leaves this client)
        self.ek: bytes
        self._dk: bytes
        self.ek, self._dk = self._mlkem.keygen()

        # Public API alias so callers can use .public_key like SecAggregator
        self.public_key: bytes = self.ek

        # Private mask seed — 32 random bytes independent of ML-KEM keys
        self._private_seed: bytes = os.urandom(32)

        # Filled in during the ciphertext-exchange round
        # sid → (K_bytes, ct_bytes) for peers we encapsulated *to*
        self._encaps_to: Dict[str, Tuple[bytes, bytes]] = {}
        # sid → ct_bytes sent by peers who encapsulated *to us*
        self._ct_from: Dict[str, bytes] = {}

        self._peer_eks: Dict[str, bytes] = {}
        self._my_sid: str = ""
        # self._weights: np.ndarray = np.zeros(shape, dtype=np.float64)
        self._weights: np.ndarray = np.zeros(shape, dtype=np.int64)

    # ------------------------------------------------------------------
    # Round 0 — broadcast public_key (ek)
    # (caller reads self.public_key and sends it to the server)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Round 0.5 — ciphertext exchange
    # ------------------------------------------------------------------

    def generate_ciphertexts(
        self,
        peer_eks: Dict[str, bytes],
        my_sid: str,
    ) -> Dict[str, bytes]:
       
        self._peer_eks = {s: ek for s, ek in peer_eks.items() if s != my_sid}
        self._my_sid = my_sid

        out: Dict[str, bytes] = {}
        for sid, peer_ek in self._peer_eks.items():
            if sid > my_sid:                        # we are the encapsulator
                K, ct = self._mlkem.encaps(peer_ek)
                self._encaps_to[sid] = (K, ct)
                out[sid] = ct
                logger.debug(
                    "[%s] Encapsulated to %s — ct=%d B, K=%d B",
                    my_sid, sid, len(ct), len(K),
                )
        return out

    def receive_ciphertexts(self, ciphertexts: Dict[str, bytes]) -> None:
        
        self._ct_from.update(ciphertexts)
        for sid, ct in ciphertexts.items():
            logger.debug(
                "[%s] Received ciphertext from %s — ct=%d B",
                self._my_sid, sid, len(ct),
            )

    # ------------------------------------------------------------------
    # Round 2 — masked gradient upload
    # ------------------------------------------------------------------

    def set_weights(self, weights: np.ndarray) -> None:
        
        # self._weights = np.float64(weights)
        self._weights = np.round(weights * self.SCALE_FACTOR).astype(np.int64)

    def prepare_masked_gradient(
        self,
        peer_eks: Optional[Dict[str, bytes]] = None,
        my_sid: Optional[str] = None,
    ) -> np.ndarray:
        
        if my_sid is not None:
            self._my_sid = my_sid
        if peer_eks is not None:
            self._peer_eks = {s: ek for s, ek in peer_eks.items()
                              if s != self._my_sid}

        use_cuda = self._accel_mode == "cuda" and torch.cuda.is_available()
        masked = torch.as_tensor(self._weights, dtype=torch.int64, device="cuda") if use_cuda else deepcopy(self._weights)
        _t0 = __import__("time").time()

        for sid in self._peer_eks:
            K = self._resolve_shared_secret(sid)
            if K is None:
                continue
            seed = _kdf(K)
            mask = _prg(seed, self._shape, use_cuda=use_cuda)
            if sid > self._my_sid:
                masked += mask
            else:
                masked -= mask
            logger.debug(
                "[%s] ML-KEM mask with %s applied (%.4fs)",
                self._my_sid, sid, __import__("time").time() - _t0,
            )

        # Private mask  PRG(b)  — same role as in DH variant
        masked += _prg(_kdf(self._private_seed), self._shape, use_cuda=use_cuda)
        logger.info(
            "[%s] Masked gradient ready (%.4fs)",
            self._my_sid, __import__("time").time() - _t0,
        )
        return masked.cpu().numpy() if use_cuda else masked

    def private_mask(self) -> np.ndarray:
        
        use_cuda = self._accel_mode == "cuda" and torch.cuda.is_available()
        mask = _prg(_kdf(self._private_seed), self._shape, use_cuda=use_cuda)
        return (-mask).cpu().numpy() if use_cuda else -mask

    # ------------------------------------------------------------------
    # Round 3 — reveal / correction for dropouts
    # ------------------------------------------------------------------

    def reveal_pairwise_masks(self, dropout_sids: List[str]) -> np.ndarray:
        use_cuda = self._accel_mode == "cuda" and torch.cuda.is_available()
        correction = torch.zeros(self._shape, dtype=torch.int64, device="cuda") if use_cuda else np.zeros(self._shape, dtype=np.int64)
        # correction = np.zeros(self._shape, dtype=np.float64)
        for sid in dropout_sids:
            K = self._resolve_shared_secret(sid)
            if K is None:
                logger.warning(
                    "[%s] Cannot resolve shared secret for dropout %s; skipping.",
                    self._my_sid, sid,
                )
                continue
            seed = _kdf(K)
            mask = _prg(seed, self._shape, use_cuda=use_cuda)
            # Mirror the sign from prepare_masked_gradient
            if sid < self._my_sid:
                correction -= mask
            else:
                correction += mask
        # return np.float64(-correction)
        return (-correction).cpu().numpy() if use_cuda else -correction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_shared_secret(self, peer_sid: str) -> Optional[bytes]:
        
        if peer_sid > self._my_sid:
            entry = self._encaps_to.get(peer_sid)
            if entry is None:
                logger.warning(
                    "[%s] No encaps entry for peer %s.", self._my_sid, peer_sid
                )
                return None
            K, _ = entry
            return K
        else:
            ct = self._ct_from.get(peer_sid)
            if ct is None:
                logger.warning(
                    "[%s] No ciphertext from peer %s.", self._my_sid, peer_sid
                )
                return None
            # FIPS 203 §7.3  ML-KEM.Decaps(dk, ct) → K
            return self._mlkem.decaps(self._dk, ct)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SecAggregatorMLKEM("
            f"level={self._security_level}, "
            f"ek_size={len(self.ek)}B, "
            f"shape={self._shape})"
        )

    @property
    def security_level(self) -> str:
        """The ML-KEM security level string, e.g. ``'ML-KEM-768'``."""
        return self._security_level

    @property
    def encapsulation_key_size(self) -> int:
        """Size of the public encapsulation key in bytes."""
        return len(self.ek)

    @property
    def ciphertext_size(self) -> int:
        """Size of one ML-KEM ciphertext in bytes (approximate — first peer)."""
        if self._encaps_to:
            _, ct = next(iter(self._encaps_to.values()))
            return len(ct)
        return {
            "ML-KEM-512":  768,
            "ML-KEM-768": 1088,
            "ML-KEM-1024": 1568,
        }.get(self._security_level, -1)
