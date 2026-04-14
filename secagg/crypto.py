

import hashlib
import logging
from copy import deepcopy
from random import randrange
from typing import Dict, List, Tuple

import numpy as np

from . import config

logger = logging.getLogger(__name__)




def _dh_seed_to_numpy_seed(shared_secret: int) -> int:
    if shared_secret <= 0xFFFF_FFFF:
        return int(shared_secret)
    raw = shared_secret.to_bytes((shared_secret.bit_length() + 7) // 8, "big")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _prg(seed: int, shape: Tuple[int, int]) -> np.ndarray:
    np.random.seed(_dh_seed_to_numpy_seed(seed))
    # return np.float64(np.random.rand(*shape))
    return np.random.randint(-10**14, 10**14, size=shape, dtype=np.int64)



# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SecAggregator:


    SCALE_FACTOR = 100000000.0

    def __init__(
        self,
        shape: Tuple[int, int] = config.MODEL_SHAPE,
        dh_generator: int = config.DH_GENERATOR,
        dh_prime: int = config.DH_PRIME,
    ) -> None:
        self._g: int = dh_generator
        self._p: int = dh_prime
        self._shape: Tuple[int, int] = shape

        # DH secret key  x ∈ [2, p−2]
        self._secret_key: int = randrange(2, self._p - 1)
        # Private mask seed  b ∈ [2, p−2]  (independent random)
        self._private_seed: int = randrange(2, self._p - 1)

        # DH public key  g^x mod p
        self.public_key: int = pow(self._g, self._secret_key, self._p)

        # Filled during prepare_masked_gradient()
        self._peer_keys: Dict[str, int] = {}   
        self._my_sid: str = ""
        # self._weights: np.ndarray = np.zeros(shape, dtype=np.float64)

        self._weights: np.ndarray = np.zeros(shape, dtype=np.int64)


    def set_weights(self, weights: np.ndarray) -> None:
        # self._weights = np.float64(weights)
        self._weights = np.round(weights * self.SCALE_FACTOR).astype(np.int64)

    def prepare_masked_gradient(
        self, peer_public_keys: Dict[str, int], my_sid: str
    ) -> np.ndarray:
        self._peer_keys = {
            sid: pk for sid, pk in peer_public_keys.items() if sid != my_sid
        }
        self._my_sid = my_sid

        masked = deepcopy(self._weights)
        t0 = __import__("time").time()

        for sid, peer_pk in self._peer_keys.items():
            shared = pow(peer_pk, self._secret_key, self._p)
            mask = _prg(shared, self._shape)
            if sid > my_sid:
                masked += mask
            else:
                masked -= mask
            logger.debug("DH secret with %s computed (%.4fs)", sid,
                         __import__("time").time() - t0)

        # Private mask PRG(b)
        masked += _prg(self._private_seed, self._shape)
        logger.info("Masked gradient ready (total mask time: %.4fs)",
                    __import__("time").time() - t0)
        return masked

    def private_mask(self) -> np.ndarray:
        return -_prg(self._private_seed, self._shape)

    def reveal_pairwise_masks(self, dropout_sids: List[str]) -> np.ndarray:
        # correction = np.zeros(self._shape, dtype=np.float64)
        correction = np.zeros(self._shape, dtype=np.int64)
        for sid in dropout_sids:
            if sid not in self._peer_keys:
                logger.warning("Dropout sid %s not in peer keys; skipping.", sid)
                continue
            shared = pow(self._peer_keys[sid], self._secret_key, self._p)
            mask = _prg(shared, self._shape)
            # Mirror of the sign used in prepare_masked_gradient
            if sid < self._my_sid:
                correction -= mask
            else:
                correction += mask
        return -correction
