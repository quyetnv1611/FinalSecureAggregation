import socketio
from random import randrange
import numpy as np
from copy import deepcopy
import codecs
import pickle
import json
import time
import hashlib

DH_PRIME_2048 = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
DH_GENERATOR = 2  # generator chuẩn g=2 của Group 14


class SecAggregator:
    def __init__(self, common_base, common_mod, dimensions, weights):
        self.secretkey = randrange(2, common_mod - 1)
        self.base = common_base
        self.mod = common_mod
        self.pubkey = pow(self.base, self.secretkey, self.mod)  # g^x mod p
        self.sndkey = randrange(2, common_mod - 1)              # private mask seed
        self.dim = dimensions
        self.weights = weights
        self.keys = {}
        self.id = ''

    def public_key(self):
        return self.pubkey

    def set_weights(self, wghts, dims):
        self.weights = wghts
        self.dim = dims

    def configure(self, base, mod):
        self.base = base
        self.mod = mod
        self.secretkey = randrange(2, mod - 1)      # re-sample với mod mới
        self.sndkey    = randrange(2, mod - 1)
        self.pubkey    = pow(self.base, self.secretkey, self.mod)  # g^x mod p

    def generate_weights(self, seed):
        if isinstance(seed, int) and seed > 0xFFFFFFFF:
            raw = seed.to_bytes((seed.bit_length() + 7) // 8, 'big')
            seed = int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')
        np.random.seed(int(seed) & 0xFFFFFFFF)
        return np.float64(np.random.rand(self.dim[0], self.dim[1]))

    def prepare_weights(self, shared_keys, myid):
        self.keys = shared_keys
        self.id = myid
        wghts = deepcopy(self.weights)
        for sid in shared_keys:
            shared_secret = pow(shared_keys[sid], self.secretkey, self.mod)
            if sid > myid:
                wghts += self.generate_weights(shared_secret)
            elif sid < myid:
                wghts -= self.generate_weights(shared_secret)
        wghts += self.generate_weights(self.sndkey)
        return wghts

    def reveal(self, keylist):
        wghts = np.zeros(self.dim)
        for each in keylist:
            shared_secret = pow(self.keys[each], self.secretkey, self.mod)
            if each < self.id:
                wghts -= self.generate_weights(shared_secret)
            elif each > self.id:
                wghts += self.generate_weights(shared_secret)
        return -1 * wghts

    def private_secret(self):
        return self.generate_weights(self.sndkey)


class secaggclient:
    def __init__(self, serverhost, serverport):
        self.sio = socketio.Client(request_timeout=60)
        self.server_url = f"http://{serverhost}:{serverport}"
        self.aggregator = SecAggregator(3, 100103, (10, 10), np.float64(np.full((10, 10), 3, dtype=int)))
        self.id = ''
        self.keys = {}
        self.send_times = {}

    def start(self):
        self.register_handles()
        print("[CLIENT] Starting and connecting to server...")
        self.sio.connect(self.server_url)
        self.sio.emit("wakeup")
        self.sio.wait()

    def configure(self, b, m):
        self.aggregator.configure(b, m)

    def set_weights(self, wghts, dims):
        self.aggregator.set_weights(wghts, dims)

    def weights_encoding(self, x):
        return codecs.encode(pickle.dumps(x), 'base64').decode()

    def weights_decoding(self, s):
        return pickle.loads(codecs.decode(s.encode(), 'base64'))

    def register_handles(self):

        @self.sio.on('connect')
        def on_connect():
            print("[CLIENT] Connected to server")

        @self.sio.on('send_public_key')
        def on_send_pubkey(msg):
            self.id = msg['id']
            pubkey_hex = format(self.aggregator.public_key(), 'x')
            pubkey = {'key': pubkey_hex}
            t0 = time.time()
            self.sio.emit('public_key', pubkey)
            t1 = time.time()

        @self.sio.on('public_keys')
        def on_sharedkeys(keys_json):
            keydict_raw = json.loads(keys_json)
            keydict = {sid: int(hex_val, 16) for sid, hex_val in keydict_raw.items()}
            self.keys = keydict
            t0 = time.time()
            weight = self.aggregator.prepare_weights(self.keys, self.id)
            t1 = time.time()
            weight_enc = self.weights_encoding(weight)
            resp = {'weights': weight_enc}
            t2 = time.time()
            self.sio.emit('weights', resp)
            t3 = time.time()

        @self.sio.on('send_secret')
        def on_send_secret(msg):
            t0 = time.time()
            secret = self.weights_encoding(-1 * self.aggregator.private_secret())
            resp = {'secret': secret}
            t1 = time.time()
            self.sio.emit('secret', resp)
            t2 = time.time()

        @self.sio.on('send_there_secret')
        def on_reveal_secret(keylist_json):
            t0 = time.time()
            keylist = json.loads(keylist_json)
            revealed = self.weights_encoding(self.aggregator.reveal(keylist))
            resp = {'rvl_secret': revealed}
            t1 = time.time()
            self.sio.emit('rvl_secret', resp)
            t2 = time.time()
            self.sio.disconnect()

        @self.sio.on('disconnect')
        def on_disconnect():
            pass

        @self.sio.on('late')
        def on_late(msg):
            self.sio.disconnect()


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    np.random.seed(seed)
    weights = np.float64(np.random.uniform(0.1, 1.0, (10, 10)))
    s = secaggclient("127.0.0.1", 2019)
    s.set_weights(weights, (10, 10))
    s.configure(DH_GENERATOR, DH_PRIME_2048)
    s.start()
