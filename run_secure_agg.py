
import subprocess
import time
import sys
import threading
import os
import numpy as np

import math

N_CLIENTS = 50
K_THRESHOLD = math.ceil(2 * N_CLIENTS / 3)   # = 7

PYTHON = r'C:/Users/quyet/AppData/Local/Programs/Python/Python313/python.exe'
SERVER_SCRIPT = 'server.py'
CLIENT_SCRIPT = 'client.py'

CLIENT_SEEDS = list(range(1, N_CLIENTS + 1))

expected_all = np.zeros((10, 10), dtype=np.float64)
for seed in CLIENT_SEEDS:
    np.random.seed(seed)
    expected_all += np.float64(np.random.uniform(0.1, 1.0, (10, 10)))

expected_k = np.zeros((10, 10), dtype=np.float64)
for seed in CLIENT_SEEDS[:K_THRESHOLD]:
    np.random.seed(seed)
    expected_k += np.float64(np.random.uniform(0.1, 1.0, (10, 10)))



env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'

def stream_output(proc, prefix):
    for line in iter(proc.stdout.readline, b''):
        text = line.decode(errors='replace').rstrip()
        if text:
            print(f"[{prefix}] {text}", flush=True)

print('Starting SERVER...')
server_proc = subprocess.Popen(
    [PYTHON, '-u', SERVER_SCRIPT],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env=env
)
threading.Thread(target=stream_output, args=(server_proc, 'SERVER'), daemon=True).start()

time.sleep(3)

client_procs = []
for i in range(N_CLIENTS):
    seed = CLIENT_SEEDS[i]
    print(f'\nStarting CLIENT {i+1} (seed={seed})...')
    proc = subprocess.Popen(
        [PYTHON, '-u', CLIENT_SCRIPT, str(seed)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env
    )
    threading.Thread(target=stream_output, args=(proc, f'CLIENT-{i+1}'), daemon=True).start()
    client_procs.append(proc)
    time.sleep(0.5)

PER_CLIENT_TIMEOUT = max(60, N_CLIENTS * 6)
print(f'\nWaiting for all clients (timeout={PER_CLIENT_TIMEOUT}s each)...')
for proc in client_procs:
    try:
        proc.wait(timeout=PER_CLIENT_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[WARN] Client timed out!", flush=True)
        proc.kill()

time.sleep(10)
server_proc.terminate()
print('\n' + '=' * 60)
print('All clients finished. Server terminated.')
