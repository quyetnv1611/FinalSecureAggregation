# SecureAggregation

An implementation of Secure Aggregation algorithm based on "Practical Secure Aggregation for Privacy-Preserving Machine Learning
(Bonawitz et. al)" in Python.

Dependencies: Flask, socketio and socketIO_client

`pip install Flask`

`pip install socketio`

`pip install socketIO-client`

# Usage:
## Client side:
### Init:

`c = secaggclient(host,port)`

### Give weights needed to be transmitted (originally set to zero)

`c.set_weights(nd_numpyarray,dimensions_of_array)`

### Set common base and mod

`c.configure(common_base, common_mod)`

### start client side:

`c.start()`

## Server side:
### init:

`s = secaggserver(host,port,n,k)`

where n is number of selected clients for the round and k is number of client responses required before aggregation process begins

### start server side:

`s.start()`

## CUDA backend

If you build a real CUDA-enabled crypto backend, the benchmark can load it directly from shared libraries.

For ML-KEM, point `SECAGG_CUDA_KEM_LIBRARY` to a CUDA build of `liboqs.so` with `OQS_USE_CUPQC=ON`.
For ML-DSA, point `SECAGG_CUDA_SIG_LIBRARY` to a shared library built from `cuDilithium`.

Example:

```bash
python experiments/benchmarks/bench_orig_vs_pq.py \
	--crypto-accel cuda \
	--cuda-kem-library /path/to/liboqs.so \
	--cuda-sig-library /path/to/libcuDilithium3.so
```

If those libraries are not present, the benchmark will fall back to CPU or fail fast when `--require-cuda-backend` is set.

## Colab setup

Use the first cell to build or upload the CUDA libraries into Google Drive. Use the second cell to run the benchmark.

### 1. Build or place the CUDA libraries in Drive

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!mkdir -p /content/drive/MyDrive/secagg_build

# Put or build these files under /content/drive/MyDrive/secagg_build
#   liboqs.so
#   libcuDilithium3.so (or libcuDilithium2.so / libcuDilithium5.so)

!find /content/drive -name 'liboqs.so' -o -name 'libcuDilithium3.so' -o -name 'libcuDilithium2.so' -o -name 'libcuDilithium5.so'
```

### 2. Run the benchmark

```python
import os
from pathlib import Path

%cd /content/

if not os.path.exists('FinalSecureAggregation'):
    !git clone https://github.com/quyetnv1611/FinalSecureAggregation

%cd /content/FinalSecureAggregation

!nvidia-smi
!python --version
!pip install -U pip
!pip install -r requirements.txt
!python -m py_compile secagg/cuda_shared_lib.py secagg/crypto_backend_plugins.py secagg/crypto_backend.py experiments/benchmarks/bench_orig_vs_pq.py

def find_first(root: str, patterns: list[str]) -> str | None:
    base = Path(root)
    for pattern in patterns:
        for path in base.rglob(pattern):
            if path.is_file():
                return str(path)
    return None

os.environ['SECAGG_CUDA_LIBRARY_ROOT'] = '/content/drive'

kem_lib = find_first('/content/drive', ['liboqs.so'])
sig_lib = find_first('/content/drive', ['libcuDilithium3.so', 'libcuDilithium2.so', 'libcuDilithium5.so'])

print('KEM library:', kem_lib)
print('SIG library:', sig_lib)

if not kem_lib or not sig_lib:
    print('Missing CUDA backend. Build or upload the .so files first, then rerun this cell.')
else:
    os.environ['SECAGG_CRYPTO_ACCEL'] = 'cuda'
    os.environ['SECAGG_CUDA_KEM_LIBRARY'] = kem_lib
    os.environ['SECAGG_CUDA_SIG_LIBRARY'] = sig_lib

    !python experiments/benchmarks/bench_orig_vs_pq.py \
        --crypto-accel cuda \
        --require-cuda-backend \
        --vector-sizes 100000,200000,300000,400000,500000 \
        --clients 100,200 \
        --dropouts 0.0,0.1 \
        --n-repeat 3 \
        --reset-checkpoint
```
