from __future__ import annotations

import argparse

from secagg.crypto_backend import configure_backend_environment, cuda_kem_available, cuda_sig_available, resolve_mode
from secagg.crypto_mlkem import SecAggregatorMLKEM
from secagg.sig_pq import make_signer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check crypto adapter wiring")
    parser.add_argument("--crypto-accel", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cuda-kem-module", default="")
    parser.add_argument("--cuda-sig-module", default="")
    parser.add_argument("--cpu-kem-module", default="")
    parser.add_argument("--cpu-sig-module", default="")
    parser.add_argument("--prefer-liboqs", action="store_true")
    args = parser.parse_args()

    configure_backend_environment(
        crypto_accel=args.crypto_accel,
        cuda_kem_module=args.cuda_kem_module or None,
        cuda_sig_module=args.cuda_sig_module or None,
        cpu_kem_module=args.cpu_kem_module or None,
        cpu_sig_module=args.cpu_sig_module or None,
        prefer_liboqs=args.prefer_liboqs,
    )

    print(f"[check] requested={args.crypto_accel} effective={resolve_mode(args.crypto_accel)}")
    print(f"[check] kem_cuda_available={cuda_kem_available()} sig_cuda_available={cuda_sig_available()}")

    a = SecAggregatorMLKEM(shape=(8,), security_level="ML-KEM-768")
    b = SecAggregatorMLKEM(shape=(8,), security_level="ML-KEM-768")
    cts = a.generate_ciphertexts({"A": a.public_key, "B": b.public_key}, "A")
    b.receive_ciphertexts({"A": cts["B"]})
    print(f"[check] KEM ok: pk_size={a.encapsulation_key_size} ct_size={len(cts['B'])}")

    signer = make_signer("ML-DSA-65")
    pk, sk = signer.keygen()
    msg = b"adapter-check"
    sig = signer.sign(sk, msg)
    print(f"[check] SIG ok: verify={signer.verify(pk, msg, sig)} sig_size={len(sig)}")


if __name__ == "__main__":
    main()
