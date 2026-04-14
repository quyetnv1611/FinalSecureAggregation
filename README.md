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

Use this cell in Colab to clone the repo, install dependencies, point to your CUDA shared libraries, and run the benchmark:

```python
import os

# Optional: mount Google Drive if your built .so files live there.
# from google.colab import drive
# drive.mount('/content/drive')

%cd /content/

if not os.path.exists('FinalSecureAggregation'):
    !git clone https://github.com/quyetnv1611/FinalSecureAggregation

%cd /content/FinalSecureAggregation

!nvidia-smi
!python --version
!pip install -U pip
!pip install -r requirements.txt
!python -m py_compile secagg/cuda_shared_lib.py secagg/crypto_backend_plugins.py secagg/crypto_backend.py experiments/benchmarks/bench_orig_vs_pq.py

os.environ['SECAGG_CRYPTO_ACCEL'] = 'cuda'

# Set these to the actual shared libraries you built or mounted in Colab.
# Example:
#   /content/drive/MyDrive/secagg_build/liboqs.so
#   /content/drive/MyDrive/secagg_build/libcuDilithium3.so
os.environ['SECAGG_CUDA_KEM_LIBRARY'] = '/content/drive/MyDrive/secagg_build/liboqs.so'
os.environ['SECAGG_CUDA_SIG_LIBRARY'] = '/content/drive/MyDrive/secagg_build/libcuDilithium3.so'

# Optional CPU fallback modules, if you want liboqs on CPU when CUDA is not available.
# os.environ['SECAGG_CPU_KEM_MODULE'] = 'oqs'
# os.environ['SECAGG_CPU_SIG_MODULE'] = 'oqs'

!python experiments/benchmarks/bench_orig_vs_pq.py \
    --crypto-accel cuda \
    --require-cuda-backend \
    --cuda-kem-library /content/drive/MyDrive/secagg_build/liboqs.so \
    --cuda-sig-library /content/drive/MyDrive/secagg_build/libcuDilithium3.so \
    --vector-sizes 100000,200000,300000,400000,500000 \
    --clients 100,200 \
    --dropouts 0.0,0.1 \
    --n-repeat 3 \
    --reset-checkpoint
```
