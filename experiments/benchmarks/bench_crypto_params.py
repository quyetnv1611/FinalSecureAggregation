"""
experiments/benchmarks/bench_crypto_params.py
===============================================
Benchmark 4 — Cryptographic primitive comparison.

Measures key sizes, signature sizes, and per-operation timings for:
  KEM  : DH-2048, ML-KEM-512, ML-KEM-768, ML-KEM-1024
  SIG  : Classic (ECDSA-P256), ML-DSA-44/65/87, SLH-DSA-shake_128f

Output
------
``results/bench_crypto_kem.csv``   — KEM results
``results/bench_crypto_sig.csv``   — SIG results
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_KEM = RESULTS_DIR / "bench_crypto_kem.csv"
OUT_SIG = RESULTS_DIR / "bench_crypto_sig.csv"

N_REPEAT = 10    # timing trials per primitive


# ---------------------------------------------------------------------------
# KEM benchmarks
# ---------------------------------------------------------------------------

def _bench_kem(backend: str, n: int = N_REPEAT) -> dict:
    """Benchmark one KEM backend, return stats dict."""
    if backend == "DH":
        from secagg.crypto import SecAggregator
        times_keygen = []
        times_agree  = []
        from secagg.config import DH_PRIME
        for _ in range(n):
            t = time.perf_counter(); a = SecAggregator(); b = SecAggregator()
            times_keygen.append(time.perf_counter() - t)
            # DH key agreement: shared = pk_b ^ sk_a mod p
            t = time.perf_counter()
            _shared = pow(b.public_key, a._secret_key, DH_PRIME)
            times_agree.append(time.perf_counter() - t)
        return {
            "backend":    "DH-2048",
            "pk_bytes":   (2048 // 8),   # 256 bytes
            "ct_bytes":   0,             # N/A for DH
            "keygen_ms":  round(sum(times_keygen) / n * 1000, 3),
            "encaps_ms":  round(sum(times_agree)  / n * 1000, 3),
            "decaps_ms":  0,
            "pq_safe":    False,
        }
    else:
        from secagg.crypto_mlkem import SecAggregatorMLKEM
        times_keygen = []
        times_encaps = []
        times_decaps = []
        for _ in range(n):
            t = time.perf_counter()
            a = SecAggregatorMLKEM(security_level=backend)
            b = SecAggregatorMLKEM(security_level=backend)
            times_keygen.append(time.perf_counter() - t)

            t = time.perf_counter()
            cts = a.generate_ciphertexts({"A": a.public_key, "B": b.public_key}, "A")
            times_encaps.append(time.perf_counter() - t)

            t = time.perf_counter()
            b.receive_ciphertexts({"A": cts["B"]})
            times_decaps.append(time.perf_counter() - t)

        return {
            "backend":    backend,
            "pk_bytes":   a.encapsulation_key_size,
            "ct_bytes":   a.ciphertext_size,
            "keygen_ms":  round(sum(times_keygen) / n * 1000, 3),
            "encaps_ms":  round(sum(times_encaps) / n * 1000, 3),
            "decaps_ms":  round(sum(times_decaps) / n * 1000, 3),
            "pq_safe":    True,
        }


# ---------------------------------------------------------------------------
# SIG benchmarks
# ---------------------------------------------------------------------------

def _bench_sig(backend: str, n: int = N_REPEAT) -> dict:
    from secagg.sig_pq import make_signer
    MSG = b"round0_key_broadcast_data"
    s = make_signer(backend)
    pk, sk = s.keygen()             # warm-up

    times_keygen = []
    times_sign   = []
    times_verify = []

    for _ in range(n):
        t = time.perf_counter(); pk, sk = s.keygen(); times_keygen.append(time.perf_counter() - t)
        t = time.perf_counter(); sig = s.sign(sk, MSG); times_sign.append(time.perf_counter() - t)
        t = time.perf_counter(); s.verify(pk, MSG, sig); times_verify.append(time.perf_counter() - t)

    pq = backend != "classic"
    return {
        "backend":    backend,
        "pk_bytes":   len(pk),
        "sk_bytes":   len(sk),
        "sig_bytes":  len(sig),
        "keygen_ms":  round(sum(times_keygen) / n * 1000, 3),
        "sign_ms":    round(sum(times_sign)   / n * 1000, 3),
        "verify_ms":  round(sum(times_verify) / n * 1000, 3),
        "pq_safe":    pq,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> None:
    # --- KEM ---
    kem_backends = ["DH", "ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
    kem_rows = []
    for b in kem_backends:
        print(f"[crypto_params] KEM  {b}")
        row = _bench_kem(b)
        kem_rows.append(row)
        print(f"  pk={row['pk_bytes']}B  ct={row['ct_bytes']}B  "
              f"keygen={row['keygen_ms']}ms  encaps={row['encaps_ms']}ms  "
              f"decaps={row['decaps_ms']}ms  pq={row['pq_safe']}")

    with OUT_KEM.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kem_rows[0]))
        w.writeheader(); w.writerows(kem_rows)
    print(f"[crypto_params] KEM results → {OUT_KEM}")

    # --- SIG ---
    sig_backends = ["classic", "ML-DSA-44", "ML-DSA-65", "ML-DSA-87",
                    "SLH-DSA-shake_128f"]
    sig_rows = []
    for b in sig_backends:
        print(f"[crypto_params] SIG  {b}")
        row = _bench_sig(b)
        sig_rows.append(row)
        print(f"  pk={row['pk_bytes']}B  sk={row['sk_bytes']}B  sig={row['sig_bytes']}B  "
              f"keygen={row['keygen_ms']}ms  sign={row['sign_ms']}ms  "
              f"verify={row['verify_ms']}ms  pq={row['pq_safe']}")

    with OUT_SIG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sig_rows[0]))
        w.writeheader(); w.writerows(sig_rows)
    print(f"[crypto_params] SIG results → {OUT_SIG}")


if __name__ == "__main__":
    run()
