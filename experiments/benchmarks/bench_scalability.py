from __future__ import annotations


"""
experiments/benchmarks/bench_scalability.py
============================================
Benchmark 2 — SecAgg protocol scalability.

Measures wall-clock time for each Bonawitz protocol phase as a function
of n_clients and threshold_rate, matching the experiment design from
Table 1 of Bonawitz et al. (CCS 2017).

Parameters swept
----------------
* n_clients    : [10, 20, 50, 100, 200]
* threshold_rate: [0.5, 0.6, 0.7, 0.8]   (k = ceil(threshold × n))
* crypto       : DH (original) and ML-KEM-768 (post-quantum)

Output
------
``results/bench_scalability.csv``

Columns: n_clients, threshold_rate, k_threshold, crypto, phase, time_sec
"""


import csv
import json
import os
import sys
import time
import gc
from pathlib import Path

# Force single-thread execution for underlying numeric runtimes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

# ĐOẠN TEST TỐI GIẢN SPHINCS+ (SLH-DSA) — đặt sau tất cả import
if __name__ == "__main__":
    print("[TEST] SPHINCSPlusSigner self-check with static message...")
    try:
        from secagg.sig_pq import SPHINCSPlusSigner
        signer = SPHINCSPlusSigner("SLH-DSA-shake_128f")
        msg = b"benchmark-secagg-signature-test"
        pk, sk = signer.keygen()
        # print(f"  pk type={type(pk)} len={len(pk)}")
        # print(f"  sk type={type(sk)} len={len(sk)}")
        sig = signer.sign(sk, msg)
        # print(f"  sig type={type(sig)} len={len(sig)}")
        ok = signer.verify(pk, msg, sig)
        print(f"  verify result: {ok}")
        if not ok:
            print(f"  pk={pk}\n  sk={sk}\n  sig={sig}\n  msg={msg}")
    except Exception as e:
        print(f"[TEST] Exception: {e}")

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "bench_scalability_sig.csv"

# Checkpoint settings for resumable runs
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUT_CHECKPOINT = CHECKPOINT_DIR / "bench_scalability_progress.json"

# Bonawitz Table 1 parameter grid
N_CLIENTS_LIST    = [50, 100, 200]
# N_CLIENTS_LIST    = [10, 20, 50, 100]
#N_CLIENTS_LIST    = [10, 20, 50, 100, 200]


THRESHOLD_RATES   = [0.5, 0.6, 0.7, 0.8]
CRYPTO_BACKENDS   = [
    ("DH", "classic"),
    # ("ML-KEM-768", "classic"),
    # ("DH", "ML-DSA-65"),
    ("ML-KEM-768", "ML-DSA-65"),
    ("ML-KEM-768", "SLH-DSA-shake_128f"),
]
GRAD_SHAPE        = (100, 100)    # 10k-parameter model
N_REPEAT          = 3             # single trial — we show trends, not variances

# Cap n for backends that are O(n²): pure-Python DH and ML-KEM encaps both
# scale quadratically with n_clients on a single machine.
DH_MAX_CLIENTS    = 200   # DH 2048-bit modular exp is too slow beyond this
MLKEM_MAX_CLIENTS = 200  # ML-KEM pure-Python encaps cap for reasonable runtime


# ---------------------------------------------------------------------------
# Per-phase timing (in-process, no network)
# ---------------------------------------------------------------------------


def _time_phases(n: int, threshold_rate: float, kem_backend: str, sig_backend: str) -> dict[str, float]:
    """Return timing dict for one (n, threshold_rate, kem_backend, sig_backend) configuration."""
    import math
    k = math.ceil(threshold_rate * n)
    sids = [f"C{i:04d}" for i in range(n)]
    survivor_sids = sids[:k]
    dropout_sids  = sids[k:]

    totals = {
        "round0_keygen":    0.0,
        "round0_sign":      0.0,
        "round05_encaps":   0.0,
        "round1_verify":    0.0,
        "round2_masking":   0.0,
        "round3_correction": 0.0,
    }

    use_mlkem = kem_backend != "DH"



    # import concurrent.futures

    # Sử dụng message tĩnh cho mọi client
    STATIC_MSG = b"benchmark-secagg-signature-test"
    for _ in range(N_REPEAT):
        # ---- Round 0: key generation ----
        t = time.perf_counter()
        if use_mlkem:
            from secagg.crypto_mlkem import SecAggregatorMLKEM
            clients = {
                sid: SecAggregatorMLKEM(shape=GRAD_SHAPE, security_level=kem_backend)
                for sid in sids
            }
        else:
            from secagg.crypto import SecAggregator
            clients = {sid: SecAggregator(shape=GRAD_SHAPE) for sid in sids}
        totals["round0_keygen"] += time.perf_counter() - t
        print(f"[LOG] n={n} backend={kem_backend}+{sig_backend} round0_keygen: {totals['round0_keygen']:.3f}s")

        # ---- Round 0: sign public key (song song, test verify ngay sau sign) ----
        t = time.perf_counter()
        sig_info = {}
        def sign_and_verify_one(sid):
            # Nếu là SPHINCS+, ánh xạ sig_backend sang đúng parameter object
            if sig_backend.startswith("SLH-DSA"):
                import slhdsa.lowlevel.slhdsa as _ll
                from slhdsa.slhdsa import SecretKey
                import slhdsa
                # Map sig_backend string to slhdsa parameter object
                param_map = {
                    "SLH-DSA-shake_128f": slhdsa.shake_128f,
                    "SLH-DSA-shake_128s": slhdsa.shake_128s,
                    "SLH-DSA-shake_192f": slhdsa.shake_192f,
                    "SLH-DSA-shake_192s": slhdsa.shake_192s,
                    "SLH-DSA-shake_256f": slhdsa.shake_256f,
                    "SLH-DSA-shake_256s": slhdsa.shake_256s,
                    "SLH-DSA-sha2_128f": slhdsa.sha2_128f,
                    "SLH-DSA-sha2_128s": slhdsa.sha2_128s,
                    "SLH-DSA-sha2_192f": slhdsa.sha2_192f,
                    "SLH-DSA-sha2_192s": slhdsa.sha2_192s,
                    "SLH-DSA-sha2_256f": slhdsa.sha2_256f,
                    "SLH-DSA-sha2_256s": slhdsa.sha2_256s,
                }
                param = param_map.get(sig_backend)
                if param is None:
                    print(f"[ERROR] Unknown SLH-DSA variant: {sig_backend}")
                    return None
                # print(f"[DEBUG][keygen] sig_backend={sig_backend}, param={param}, type={type(param)}, id={id(param)}")
                msg = STATIC_MSG
                sk_tuple, pk_tuple = _ll.keygen(param)
                sk_obj = SecretKey(sk_tuple, param)
                pk_obj = sk_obj.pubkey
                # print(f"[DEBUG][sign] param={param}, type={type(param)}, id={id(param)}")
                sig = sk_obj.sign_pure(msg)
                # print(f"[DEBUG][verify] param={param}, type={type(param)}, id={id(param)}")
                ok = pk_obj.verify_pure(msg, sig)
                if not ok:
                    print(f"[ERROR] SIGN/VERIFY failed for client {sid} (self-check)")
                    # print(f"  pk_obj={pk_obj}\n  msg={msg}\n  sig={sig}")
                    return None
                return sid, (pk_obj, sig, msg, param)
            else:
                
                from secagg.sig_pq import make_signer
                local_signer = make_signer(sig_backend)
                
                sig_pk, sig_sk = local_signer.keygen()
                msg = STATIC_MSG
                sig = local_signer.sign(sig_sk, msg)
                ok = local_signer.verify(sig_pk, msg, sig)
                if not ok:
                    print(f"[ERROR] SIGN/VERIFY failed for client {sid} (self-check)")
                    # print(f"  sig_pk={sig_pk}\n  msg={msg}\n  sig={sig}")
                    return None
                return sid, (sig_pk, sig, msg)
        # if sig_backend.startswith("SLH-DSA"):
        #     results = [sign_and_verify_one(sid) for sid in sids]
        # else:
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         results = list(executor.map(sign_and_verify_one, sids))
        results = [sign_and_verify_one(sid) for sid in sids]
        for res in results:
            if res is not None:
                sid, tup = res
                sig_info[sid] = tup
        totals["round0_sign"] += time.perf_counter() - t
        print(f"[LOG] n={n} backend={kem_backend}+{sig_backend} round0_sign: {totals['round0_sign']:.3f}s")

        # ---- Round 0.5: encapsulation (ML-KEM only) ----
        t = time.perf_counter()
        if use_mlkem:
            peer_eks = {sid: c.public_key for sid, c in clients.items()}
            all_cts: dict = {}
            for sid, cli in clients.items():
                all_cts[sid] = cli.generate_ciphertexts(peer_eks, sid)
            for i, sid_v in enumerate(sids):
                incoming = {
                    sid_u: all_cts[sid_u][sid_v]
                    for sid_u in sids[:i]
                    if sid_v in all_cts.get(sid_u, {})
                }
                if incoming:
                    clients[sid_v].receive_ciphertexts(incoming)
        totals["round05_encaps"] += time.perf_counter() - t

        # ---- Round 1: verify each client's pk signature (single-threaded for stability) ----
        t = time.perf_counter()
       
        def verify_one(sid):
            if sid not in sig_info:
                return False
            tup = sig_info[sid]
            
           
            if sig_backend.startswith("SLH-DSA"):
                if len(tup) == 4:
                    sig_pk, sig, msg, param = tup
                else:
                    sig_pk, sig, msg = tup
                    param = None
                # print(f"[DEBUG][verify_one] sig_backend={sig_backend}, param={param}, type={type(param)}, id={id(param)}")
                # Thực hiện verify lại bằng pk_obj.verify_pure
                ok = sig_pk.verify_pure(msg, sig)
            else:
                from secagg.sig_pq import make_signer
                local_signer = make_signer(sig_backend) 
                
                sig_pk, sig, msg = tup
                ok = local_signer.verify(sig_pk, msg, sig)
                
            if not ok:
                print(f"[ERROR] SIG verify failed for client {sid}")
            return ok
        for sid in sig_info.keys():
            verify_one(sid)
        totals["round1_verify"] += time.perf_counter() - t

        # ---- Round 2: masked gradient upload (survivors only) ----
        t = time.perf_counter()
        masked = {}
        if use_mlkem:
            for sid in survivor_sids:
                clients[sid].set_weights(np.ones(GRAD_SHAPE, dtype=np.float64))
                masked[sid] = clients[sid].prepare_masked_gradient()
        else:
            all_pks = {sid: c.public_key for sid, c in clients.items()}
            for sid in survivor_sids:
                clients[sid].set_weights(np.ones(GRAD_SHAPE, dtype=np.float64))
                masked[sid] = clients[sid].prepare_masked_gradient(all_pks, sid)
        totals["round2_masking"] += time.perf_counter() - t

        # ---- Round 3: dropout correction ----
        t = time.perf_counter()
        for sid in survivor_sids:
            clients[sid].reveal_pairwise_masks(dropout_sids)
        totals["round3_correction"] += time.perf_counter() - t

    return {k: v / N_REPEAT for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> set[tuple]:
    """Load checkpoint from JSON. Returns set of (n, tr, kem_backend, sig_backend) tuples."""
    if not OUT_CHECKPOINT.exists():
        return set()
    
    try:
        with OUT_CHECKPOINT.open("r", encoding="utf-8") as f:
            data = json.load(f)
        completed = set(tuple(item) for item in data.get("completed", []))
        print(f"[scalability] Loaded checkpoint: {len(completed)} configs completed")
        return completed
    except Exception as e:
        print(f"[scalability] Warning: Failed to load checkpoint: {e}")
        return set()


def _save_checkpoint(completed: set[tuple]) -> None:
    """Save checkpoint to JSON."""
    try:
        data = {
            "completed": [list(item) for item in completed],
            "timestamp": time.time(),
        }
        with OUT_CHECKPOINT.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[scalability] Warning: Failed to save checkpoint: {e}")


# ---------------------------------------------------------------------------

_PHASES = (
    "round0_keygen", "round0_sign", "round05_encaps",
    "round1_verify", "round2_masking", "round3_correction"
)


def skip_phases(rows: list, n: int, tr: float, k: int, kem_backend: str, sig_backend: str) -> None:
    """Append NaN rows for all phases when a config is skipped."""
    for phase in _PHASES:
        rows.append({
            "n_clients": n, "threshold_rate": tr, "k_threshold": k,
            "kem_backend": kem_backend, "sig_backend": sig_backend, "phase": phase, "time_sec": None,
        })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> None:
    import math
    rows = []
    
    # Load checkpoint to support resumable runs
    completed = _load_checkpoint()
    total_configs = len(N_CLIENTS_LIST) * len(THRESHOLD_RATES) * len(CRYPTO_BACKENDS)
    print(f"[scalability] Total configs: {total_configs}, already completed: {len(completed)}")

    # Write header if file doesn't exist
    if not OUT_CSV.exists():
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "n_clients", "threshold_rate", "k_threshold",
                "kem_backend", "sig_backend", "phase", "time_sec"
            ])
            w.writeheader()

    for n in N_CLIENTS_LIST:
        for tr in THRESHOLD_RATES:
            for kem_backend, sig_backend in CRYPTO_BACKENDS:
                config_key = (n, tr, kem_backend, sig_backend)
                
                # Skip if already completed
                if config_key in completed:
                    print(f"[scalability] Skipping (already done): n={n:>4} threshold={tr} kem={kem_backend} sig={sig_backend}")
                    continue
                
                k = math.ceil(tr * n)
                # Cap O(n²) backends to keep runtime feasible
                if kem_backend == "DH" and n > DH_MAX_CLIENTS:
                    # print(f"[scalability] n={n:>4}  threshold={tr}  k={k}  kem={kem_backend} sig={sig_backend}  => SKIP (DH too slow for n>{DH_MAX_CLIENTS})")
                    skip_phases(rows, n, tr, k, kem_backend, sig_backend)
                    completed.add(config_key)
                    _save_checkpoint(completed)
                    continue
                if kem_backend.startswith("ML-KEM") and n > MLKEM_MAX_CLIENTS:
                    # print(f"[scalability] n={n:>4}  threshold={tr}  k={k}  kem={kem_backend} sig={sig_backend}  => SKIP (pure-Python KEM too slow for n>{MLKEM_MAX_CLIENTS})")
                    skip_phases(rows, n, tr, k, kem_backend, sig_backend)
                    completed.add(config_key)
                    _save_checkpoint(completed)
                    continue

                try:
                    print(f"[scalability] n={n:>4}  threshold={tr}  k={k}  kem={kem_backend} sig={sig_backend}")
                    timings = _time_phases(n, tr, kem_backend, sig_backend)
                    
                    # Write results to CSV
                    with OUT_CSV.open("a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=[
                            "n_clients", "threshold_rate", "k_threshold",
                            "kem_backend", "sig_backend", "phase", "time_sec"
                        ])
                        for phase, t in timings.items():
                            w.writerow({
                                "n_clients":      n,
                                "threshold_rate": tr,
                                "k_threshold":    k,
                                "kem_backend":    kem_backend,
                                "sig_backend":    sig_backend,
                                "phase":          phase,
                                "time_sec":       round(t, 6) if isinstance(t, (int, float)) and t is not None else None,
                            })
                    
                    # Mark as completed and save checkpoint
                    completed.add(config_key)
                    _save_checkpoint(completed)
                    
                except Exception as e:
                    print(f"[scalability] ERROR: n={n} threshold={tr} kem={kem_backend} sig={sig_backend}: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    # Save progress before exiting
                    _save_checkpoint(completed)
                    print(f"[scalability] Checkpoint saved before exit. Restart to resume.", file=sys.stderr)
                    raise
                
                gc.collect()
    
    # All configs completed: clean up checkpoint
    if OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[scalability] Checkpoint removed (all configs completed)")

    print(f"\n[scalability] Results saved → {OUT_CSV}")


if __name__ == "__main__":
    run()
