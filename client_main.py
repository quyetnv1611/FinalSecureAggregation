

import logging
import sys

import numpy as np

from secagg import SecAggClient
from secagg.config import DH_GENERATOR, DH_PRIME, MODEL_SHAPE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.1, 1.0, MODEL_SHAPE).astype(np.float64)
    logger.info("Client seed=%d  weights[:3]=%s", seed, weights.flat[:3])

    SecAggClient(
        weights=weights,
        dh_generator=DH_GENERATOR,
        dh_prime=DH_PRIME,
    ).run()
