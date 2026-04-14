

import codecs
import json
import logging
import pickle
import time
from typing import List, Optional, Tuple

import numpy as np
import socketio as sio_lib

from . import config
from .crypto import SecAggregator

logger = logging.getLogger(__name__)


class SecAggClient:


    def __init__(
        self,
        host: str = config.HOST,
        port: int = config.PORT,
        weights: Optional[np.ndarray] = None,
        dh_generator: int = config.DH_GENERATOR,
        dh_prime: int = config.DH_PRIME,
    ) -> None:
        self._server_url = f"http://{host}:{port}"
        self._sio = sio_lib.Client(logger=False, engineio_logger=False,
                                   request_timeout=60)

        shape = config.MODEL_SHAPE
        if weights is None:
            weights = np.zeros(shape, dtype=np.float64)

        self._crypto = SecAggregator(
            shape=shape,
            dh_generator=dh_generator,
            dh_prime=dh_prime,
        )
        self._crypto.set_weights(weights)

        self._my_sid: str = ""
        self._register_handlers()




    @staticmethod
    def _encode(array: np.ndarray) -> str:
        return codecs.encode(pickle.dumps(array), "base64").decode()


    @staticmethod
    def _decode(payload: str) -> np.ndarray:
        return pickle.loads(codecs.decode(payload.encode(), "base64"))



    def _register_handlers(self) -> None:
        sio = self._sio

        @sio.on("connect")
        def on_connect() -> None:
            logger.debug("Connected to server.")

        @sio.on("disconnect")
        def on_disconnect() -> None:
            logger.debug("Disconnected from server (sid=%s).", self._my_sid)

        # ── Round 0: receive session-id ───────────────────────────────
        @sio.on("send_public_key")
        def on_send_public_key(data: dict) -> None:
            self._my_sid = data["id"]
            # Transmit public key as hex string — avoids JSON int-overflow
            # for 2 048-bit values.
            hex_pk = format(self._crypto.public_key, "x")
            t0 = time.time()
            sio.emit("public_key", {"key": hex_pk})
            logger.info(
                "[%s] Public key sent (%d bits, %.4fs)",
                self._my_sid, len(hex_pk) * 4, time.time() - t0,
            )

        # ── Round 1: receive key directory → compute masked gradient ──
        @sio.on("public_keys")
        def on_public_keys(keys_json: str) -> None:
            # Server delivers hex-encoded public keys; decode to int.
            raw: dict = json.loads(keys_json)
            peer_keys = {sid: int(hex_val, 16) for sid, hex_val in raw.items()}

            logger.info("[%s] Received %d public keys.", self._my_sid, len(peer_keys))

            t0 = time.time()
            masked = self._crypto.prepare_masked_gradient(peer_keys, self._my_sid)
            t1 = time.time()

            sio.emit("weights", {"weights": self._encode(masked)})
            logger.info(
                "[%s] Masked gradient uploaded (%.4fs mask | %.4fs total).",
                self._my_sid, t1 - t0, time.time() - t0,
            )

        # ── Round 2a: send private mask removal ───────────────────────
        @sio.on("send_secret")
        def on_send_secret(data: dict) -> None:
            t0 = time.time()
            sio.emit("secret", {"secret": self._encode(self._crypto.private_mask())})
            logger.info("[%s] Private mask sent (%.4fs).",
                        self._my_sid, time.time() - t0)

        # ── Round 2b: send pairwise-mask correction for dropouts ──────
        @sio.on("send_there_secret")
        def on_send_there_secret(dropout_json: str) -> None:
            dropouts: List[str] = json.loads(dropout_json)
            t0 = time.time()
            correction = self._crypto.reveal_pairwise_masks(dropouts)
            sio.emit("rvl_secret", {"rvl_secret": self._encode(correction)})
            logger.info(
                "[%s] Reveal correction for %d dropouts sent (%.4fs).",
                self._my_sid, len(dropouts), time.time() - t0,
            )
            # Protocol complete — disconnect cleanly.
            sio.disconnect()

        # ── Late-arrival notice ───────────────────────────────────────
        @sio.on("late")
        def on_late(data: dict) -> None:
            logger.warning("[%s] Arrived too late; disconnecting.", self._my_sid)
            sio.disconnect()



    def run(self) -> None:
        logger.info("Connecting to %s …", self._server_url)
        self._sio.connect(self._server_url)
        self._sio.emit("wakeup")
        self._sio.wait()
        logger.info("[%s] Protocol complete.", self._my_sid)
