

from flask import Flask, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
sio = SocketIO(app)


@sio.on("connect")
def on_connect():
    print(f"[TEST-SERVER] CLIENT CONNECTED: {request.sid}", flush=True)
    emit("hello", {"msg": "hi from server"})


@sio.on("ping_server")
def on_ping(data):
    print(f"[TEST-SERVER] Received ping from {request.sid}: {data}", flush=True)
    emit("pong", {"reply": "pong!"})


if __name__ == "__main__":
    print("[TEST-SERVER] Starting on port 2020...", flush=True)
    sio.run(app, host="127.0.0.1", port=2020)
