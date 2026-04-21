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

Full CUDA for both KEM and SIG requires cuPQC SDK. If cuPQC is missing, you can still run CUDA SIG only (KEM on CPU).

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

### 1.1 Locate cuPQC SDK (required for ML-KEM CUDA)

```python
from pathlib import Path

# Auto-detect cuPQCConfig.cmake anywhere on Drive.
matches = list(Path('/content/drive').rglob('cuPQCConfig.cmake'))
if not matches:
    print('cuPQC SDK not found on Drive.')
    print('Full CUDA (KEM + SIG) is not available yet.')
    print('You can continue with CUDA SIG only, or upload/install cuPQC SDK first.')
    CUPQC_DIR = None
else:
    CUPQC_DIR = str(matches[0].parent)
    print('Found cuPQCConfig.cmake at:', str(matches[0]))
    print('Using cuPQC_DIR =', CUPQC_DIR)
```

### 1.2 Build liboqs with cuPQC (full CUDA KEM)

```python
import os

if CUPQC_DIR is None:
    print('Skip KEM CUDA build because cuPQC is missing.')
else:
    %cd /content
    if not os.path.exists('/content/liboqs'):
        !git clone https://github.com/open-quantum-safe/liboqs.git

    %cd /content/liboqs
    !rm -rf build
    !cmake -S . -B build -GNinja \
        -DOQS_BUILD_ONLY_LIB=ON \
        -DOQS_USE_OPENSSL=OFF \
        -DOQS_USE_CUPQC=ON \
        -DcuPQC_DIR="$CUPQC_DIR" \
        -DCMAKE_BUILD_TYPE=Release
    !cmake --build build -j
    !find build -name 'liboqs.so' -exec cp {} /content/drive/MyDrive/secagg_build/ \;
    !find /content/drive/MyDrive/secagg_build -name 'liboqs.so' -print
```

### 1.3 Build cuDilithium shared library (CUDA SIG)

```python
import os

%cd /content
if not os.path.exists('/content/cuDilithium'):
    !git clone https://github.com/encryptorion-lab/cuDilithium.git

%cd /content/cuDilithium
!rm -rf build
!cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
!cmake --build build -j
!find build -name 'libcuDilithium*.so' -exec cp {} /content/drive/MyDrive/secagg_build/ \;
!find /content/drive/MyDrive/secagg_build -name 'libcuDilithium*.so' -print
```

### 2. Bootstrap toolchain on Colab

```python
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

%cd /content
!apt-get update
!apt-get install -y build-essential cmake ninja-build pkg-config git
!which cmake
!which ninja
!which gcc
!which g++
!which nvcc || true
!nvidia-smi
```

### 3. Run the benchmark (auto full CUDA or SIG-only)

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

if not sig_lib:
    raise RuntimeError('No CUDA SIG library found. Build cell 1.3 first.')

os.environ['SECAGG_CRYPTO_ACCEL'] = 'cuda'
os.environ['SECAGG_CUDA_SIG_LIBRARY'] = sig_lib

if kem_lib:
    print('Mode: FULL CUDA (KEM + SIG)')
    os.environ['SECAGG_CUDA_KEM_LIBRARY'] = kem_lib
    !python experiments/benchmarks/bench_orig_vs_pq.py \
        --crypto-accel cuda \
        --require-cuda-backend \
        --require-full-cuda-backend \
        --vector-sizes 100000,200000,300000,400000,500000 \
        --clients 100,200 \
        --dropouts 0.0,0.1 \
        --n-repeat 3 \
        --reset-checkpoint
else:
    print('Mode: CUDA SIG only (KEM CPU fallback)')
    !python experiments/benchmarks/bench_orig_vs_pq.py \
        --crypto-accel cuda \
        --vector-sizes 100000,200000,300000,400000,500000 \
        --clients 100,200 \
        --dropouts 0.0,0.1 \
        --n-repeat 3 \
        --reset-checkpoint
```

## MNIST accuracy study (Table/Figure style)

Use this command on Colab to generate the three study groups like your target figure:

```bash
python experiments/benchmarks/bench_accuracy.py \
    --mnist-studies \
    --mnist-iid-mode both \
    --device cuda \
    --studies local_params,clients,dropout \
    --milestones 5,10,20,50,100 \
    --local-grid 10:1,10:5,10:20,50:1,50:5,50:20 \
    --clients-grid 5,10,20,50,100 \
    --dropout-grid 0.0,0.05,0.1,0.3,0.5 \
    --study-fixed-clients 50 \
    --study-fixed-batch 10 \
    --study-fixed-epochs 5 \
    --study-fixed-dropout 0.0 \
    --study-lr 0.01 \
    --seed 42
```

Outputs:
- `results/bench_accuracy_mnist_studies.csv`
- `figures/bench_accuracy_mnist_local_params.png`
- `figures/bench_accuracy_mnist_clients.png`
- `figures/bench_accuracy_mnist_dropout.png`

Quick smoke test (faster):

```bash
python experiments/benchmarks/bench_accuracy.py \
    --mnist-studies \
    --mnist-iid-mode both \
    --device cuda \
    --studies local_params \
    --milestones 5,10,20 \
    --local-grid 10:1,50:5 \
    --study-fixed-clients 20 \
    --seed 42
```

If `--device cuda` fails, it means the active Python environment does not see a CUDA-enabled PyTorch build yet. On Colab Pro, check `torch.cuda.is_available()` after installing dependencies; if it is `False`, reinstall PyTorch with a CUDA wheel or restart the runtime with GPU enabled.
