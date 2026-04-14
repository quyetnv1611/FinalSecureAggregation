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
