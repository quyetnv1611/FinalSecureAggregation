

import codecs
import json
import logging
import pickle
import time
from typing import Dict, List, Set

import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit

from . import config

logger = logging.getLogger(__name__)


class SecAggServer:

    SCALE_FACTOR = 100000000.0

    def __init__(
        self,
        host: str = config.HOST,
        port: int = config.PORT,
        n_clients: int = config.N_CLIENTS,
        k_threshold: int = config.K_THRESHOLD,
        model_shape: tuple = config.MODEL_SHAPE,
    ) -> None:
        self.host = host
        self.port = port
        self.n = n_clients
        self.k = k_threshold
        self.shape = model_shape

        # ── Aggregation state ──────────────────────────────────────────────
        # self._aggregate: np.ndarray = np.zeros(self.shape, dtype=np.float64)
        self._aggregate: np.ndarray = np.zeros(self.shape, dtype=np.int64)
        self._start_time: float | None = None
        self._end_time: float | None = None

        # Round-1 state: DH key collection
        self._client_pubkeys: Dict[str, str] = {}  # sid → hex public key
        self._n_keys_received: int = 0

        # Round-2 state: masked-gradient collection
        self._connected_sids: Set[str] = set()
        self._responded_sids: List[str] = []  # clients that sent weights in time
        self._late_sids: Set[str] = set()     # clients that were too late
        self._n_weights: int = 0

        # Round-3 state: unmasking
        self._n_secret: int = 0        # private-mask responses
        self._n_reveal: int = 0        # pairwise-mask reveal responses

        # ── Flask / SocketIO setup ─────────────────────────────────────────
        self._app = Flask(__name__)
        self._app.logger.setLevel(logging.WARNING)
        self._sio = SocketIO(self._app, logger=False, engineio_logger=False)
        self._register_handlers()




    @staticmethod
    def _encode(array: np.ndarray) -> str:
        return codecs.encode(pickle.dumps(array), "base64").decode()


    @staticmethod
    def _decode(payload: str) -> np.ndarray:
        return pickle.loads(codecs.decode(payload.encode(), "base64"))

    def _try_finalise(self) -> None:
        if self._n_secret == self.k and self._n_reveal == self.k:
            self._end_time = time.time()
            elapsed = self._end_time - (self._start_time or self._end_time)
            logger.info("===== FINAL WEIGHTS =====\n%s", self._aggregate)
            logger.info("===== SECURE AGGREGATION SUCCESSFUL! =====")
            logger.info("Total time: %.4f seconds", elapsed)
            # Also print to stdout so the runner script can capture it
            print(f"\n[SERVER] ===== FINAL WEIGHTS =====\n{self._aggregate}")
            print(f"[SERVER] ===== SECURE AGGREGATION SUCCESSFUL! =====")
            print(f"[SERVER] Total time: {elapsed:.4f} seconds\n", flush=True)



    def _register_handlers(self) -> None:  # noqa: C901  (complexity OK here)
        sio = self._sio

        @sio.on("connect")
        def on_connect() -> None:
            sid = request.sid
            self._connected_sids.add(sid)
            logger.debug("Client connected: %s  (total: %d)", sid,
                         len(self._connected_sids))

        @sio.on("disconnect")
        def on_disconnect() -> None:
            sid = request.sid
            self._connected_sids.discard(sid)
            logger.debug("Client disconnected: %s", sid)

        # ── Round 0: assign session-id ─────────────────────────────────
        @sio.on("wakeup")
        def on_wakeup() -> None:
            sid = request.sid
            if self._start_time is None:
                self._start_time = time.time()
                logger.info("Aggregation STARTED  (t=%.3f)", self._start_time)
            emit("send_public_key", {"id": sid})
            logger.debug("Assigned sid %s", sid)

        # ── Round 1: collect public keys ───────────────────────────────
        @sio.on("public_key")
        def on_public_key(data: dict) -> None:
            sid = request.sid
            hex_key: str = data["key"]
            self._client_pubkeys[sid] = hex_key
            self._n_keys_received += 1
            logger.debug(
                "Public key from %s (%d bits)  [%d/%d]",
                sid, len(hex_key) * 4, self._n_keys_received, self.n,
            )
            if self._n_keys_received == self.n:
                key_json = json.dumps(self._client_pubkeys)
                for cid in list(self._connected_sids):
                    emit("public_keys", key_json, room=cid)
                logger.info("All %d public keys broadcast.", self.n)

        # ── Round 2: collect masked gradients ─────────────────────────
        @sio.on("weights")
        def on_weights(data: dict) -> None:
            sid = request.sid
            t0 = time.time()

            if self._n_weights < self.k:
                gradient = self._decode(data["weights"])
                self._aggregate += gradient
                self._n_weights += 1
                self._responded_sids.append(sid)
                # Track late / absent clients for reveal phase
                self._late_sids = (
                    self._connected_sids
                    - set(self._responded_sids)
                    - set(self._client_pubkeys.keys())
                )
                emit("send_secret", {"msg": "send your secret"})
                logger.info(
                    "Gradient from %s decoded in %.4fs  [%d/%d]",
                    sid, time.time() - t0, self._n_weights, self.k,
                )
            else:
                self._n_weights += 1
                emit("late", {"msg": "too late"})
                logger.info("Client %s arrived late.", sid)

            if self._n_weights == self.k:
                # Clients that sent keys but NOT gradients are dropouts
                respset = set(self._client_pubkeys.keys()) - set(self._responded_sids)
                absent_json = json.dumps(list(respset))
                for cid in self._responded_sids:
                    emit("send_there_secret", absent_json, room=cid)
                logger.info(
                    "%d gradients received. Unmask phase started. "
                    "Absent clients: %s",
                    self.k, list(respset),
                )

        # ── Round 3a: private-mask removal ────────────────────────────
        @sio.on("secret")
        def on_secret(data: dict) -> None:
            sid = request.sid
            t0 = time.time()
            mask = self._decode(data["secret"])
            self._aggregate += mask
            self._n_secret += 1
            logger.info(
                "Private mask from %s decoded in %.4fs  [%d/%d]",
                sid, time.time() - t0, self._n_secret, self.k,
            )
            self._try_finalise()

        # ── Round 3b: pairwise-mask reveal ────────────────────────────
        @sio.on("rvl_secret")
        def on_rvl_secret(data: dict) -> None:
            sid = request.sid
            t0 = time.time()
            correction = self._decode(data["rvl_secret"])
            self._aggregate += correction
            self._n_reveal += 1
            logger.info(
                "Reveal correction from %s decoded in %.4fs  [%d/%d]",
                sid, time.time() - t0, self._n_reveal, self.k,
            )
            self._try_finalise()



    def run(self) -> None:
        logger.info(
            "SecAggServer listening on %s:%d  (n=%d, k=%d)",
            self.host, self.port, self.n, self.k,
        )
        print(
            f"[SERVER] n={self.n}, k={self.k}  "
            f"(Bonawitz CCS 2017: k >= ceil(2n/3))",
            flush=True,
        )
        self._sio.run(self._app, host=self.host, port=self.port)

    def get_final_float_weights(self, n_survivors: int) -> np.ndarray:
        # Chia trung bình, sau đó chia cho 100,000 để lùi dấu phẩy lại
        avg_int = self._aggregate / n_survivors
        return (avg_int / self.SCALE_FACTOR).astype(np.float32)   
