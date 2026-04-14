
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from secagg.sig_pq import make_signer, MSG


BACKENDS = [
    ("classic", "ECDSA-P256"),
    ("ML-DSA-44", "ML-DSA-44"),
    ("ML-DSA-65", "ML-DSA-65"),
    ("ML-DSA-87", "ML-DSA-87"),
    ("SLH-DSA-shake_128f", "SLH-DSA-shake_128f"),
]
MSG_BAD = b"tampered message"

if __name__ == "__main__":
    print(f"\n{'Backend':<28}  keygen  sign  verify  tamper  wrong_key")
    print("-" * 70)
    all_pass = True
    for backend_id, label in BACKENDS:
        try:
            s = make_signer(backend_id)
            pk, sk = s.keygen()
            sig = s.sign(sk, MSG)
            ok_correct = s.verify(pk, MSG, sig)
            ok_tampered = s.verify(pk, MSG_BAD, sig)
            pk2, _ = s.keygen()
            ok_wrong_key = s.verify(pk2, MSG, sig)
            passed = ok_correct and not ok_tampered and not ok_wrong_key
            if not passed:
                all_pass = False
            status = "PASS" if passed else "FAIL"
            print(
                f"{label:<28}  {'OK':>5}  {'OK':>4}  "
                f"{'OK' if ok_correct else 'FAIL':>6}  "
                f"{'rej' if not ok_tampered else 'FAIL':>6}  "
                f"{'rej' if not ok_wrong_key else 'FAIL':>8}  "
                f"=> {status}"
            )
        except Exception as exc:
            all_pass = False
            print(f"{label:<28}  ERROR: {exc}")

