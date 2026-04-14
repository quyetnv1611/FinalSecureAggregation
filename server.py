from flask import *
from flask_socketio import SocketIO, emit
from flask_socketio import *
import json
import codecs
import pickle
import numpy as np
import time


class secaggserver:
    def __init__(self, host, port, n, k):
        self.n = n
        self.k = k
        self.aggregate = np.zeros((10, 10))
        self.host = host
        self.port = port
        self.numkeys = 0
        self.responses = 0
        self.secretresp = 0
        self.othersecretresp = 0
        self.respset = set()
        self.resplist = []
        self.ready_client_ids = set()
        self.client_keys = dict()
        self.start_time = None
        self.end_time = None
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.register_handles()

    def weights_encoding(self, x):
        return codecs.encode(pickle.dumps(x), 'base64').decode()

    def weights_decoding(self, s):
        return pickle.loads(codecs.decode(s.encode(), 'base64'))

    def register_handles(self):

        @self.socketio.on("connect")
        def handle_connect():
            print(f"[SERVER] Client {request.sid} connected")
            self.ready_client_ids.add(request.sid)
            print(f"[SERVER] Connected devices: {self.ready_client_ids}")

        @self.socketio.on("wakeup")
        def handle_wakeup():
            if self.start_time is None:
                self.start_time = time.time()
                print(f"[SERVER] ===== Aggregation process STARTED at {self.start_time:.6f} =====")
            print(f"[SERVER] Received wakeup from {request.sid}")
            emit("send_public_key", {
                "message": "hey I'm server",
                "id": request.sid
            })

        @self.socketio.on('public_key')
        def handle_pubkey(key):
            t0 = time.time()
            hex_key = key['key']
            print(f"[SERVER] Client {request.sid} sent public key ({len(hex_key)*4} bits): {hex_key[:16]}…")
            self.client_keys[request.sid] = hex_key  # lưu hex string
            self.numkeys += 1
            self.respset.add(request.sid)
            print(f"[SERVER] Public keys collected: {self.numkeys}/{self.n}")
            if self.numkeys == self.n:
                print(f"[SERVER] All public keys received. Broadcasting to all clients...")
                key_json = json.dumps(self.client_keys)
                for rid in list(self.ready_client_ids):
                    emit('public_keys', key_json, room=rid)
                t1 = time.time()
                print(f"[SERVER] Public key broadcast done. Time: {t1-t0:.6f} seconds")

        @self.socketio.on('weights')
        def handle_weights(data):
            t0 = time.time()
            print(f"[SERVER] Client {request.sid} sent masked weights")
            if self.responses < self.k:
                decoded_weights = self.weights_decoding(data['weights'])
                t1 = time.time()
                print(f"[SERVER] Decoded weights from {request.sid} in {t1-t0:.6f}s:\n{decoded_weights}")
                self.aggregate += decoded_weights
                print(f"[SERVER] Current aggregate:\n{self.aggregate}")
                emit('send_secret', {'msg': "send your secret"})
                print(f"[SERVER] Sent secret request to {request.sid}")
                self.responses += 1
                self.respset.remove(request.sid)
                self.resplist.append(request.sid)
            else:
                emit('late', {'msg': "too late"})
                self.responses += 1
            if self.responses == self.k:
                print(f"[SERVER] {self.k} weights received. Starting unmask phase...")
                absentkeyjson = json.dumps(list(self.respset))
                for rid in self.resplist:
                    emit('send_there_secret', absentkeyjson, room=rid)

        @self.socketio.on('secret')
        def handle_secret(data):
            t0 = time.time()
            print(f"[SERVER] Client {request.sid} sent SECRET (private mask)")
            decoded_secret = self.weights_decoding(data['secret'])
            t1 = time.time()
            print(f"[SERVER] Decoded secret from {request.sid} in {t1-t0:.6f}s:\n{decoded_secret}")
            self.aggregate += decoded_secret
            print(f"[SERVER] Current aggregate:\n{self.aggregate}")
            self.secretresp += 1
            if self.secretresp == self.k and self.othersecretresp == self.k:
                self.end_time = time.time()
                print(f"\n[SERVER] ===== FINAL WEIGHTS =====\n{self.aggregate}")
                print(f"[SERVER] ===== SECURE AGGREGATION SUCCESSFUL! =====")
                print(f"[SERVER] Ended at {self.end_time:.6f}")
                print(f"[SERVER] Total time: {self.end_time - self.start_time:.4f} seconds\n")

        @self.socketio.on('rvl_secret')
        def handle_secret_reveal(data):
            t0 = time.time()
            print(f"[SERVER] Client {request.sid} sent revealed mask (for dropped clients)")
            decoded_reveal = self.weights_decoding(data['rvl_secret'])
            t1 = time.time()
            print(f"[SERVER] Decoded revealed mask from {request.sid} in {t1-t0:.6f}s:\n{decoded_reveal}")
            self.aggregate += decoded_reveal
            print(f"[SERVER] Current aggregate:\n{self.aggregate}")
            self.othersecretresp += 1
            if self.secretresp == self.k and self.othersecretresp == self.k:
                self.end_time = time.time()
                print(f"\n[SERVER] ===== FINAL WEIGHTS =====\n{self.aggregate}")
                print(f"[SERVER] ===== SECURE AGGREGATION SUCCESSFUL! =====")
                print(f"[SERVER] Ended at {self.end_time:.6f}")
                print(f"[SERVER] Total time: {self.end_time - self.start_time:.4f} seconds\n")

    def run(self):
        print(f"listening on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    import math
    N_CLIENTS = 50
    K_THRESHOLD = math.ceil(2 * N_CLIENTS / 3)
    s = secaggserver("127.0.0.1", 2019, N_CLIENTS, K_THRESHOLD)
    s.run()
