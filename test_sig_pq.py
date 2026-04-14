"""
test_sig_pq.py — Unit tests for secagg/sig_pq.py
=================================================
Tests the configurable digital signature layer.

Run:
    import sys
    sys.path.insert(0, ".")

MSG_BAD  = b"tampered_message"

BACKENDS = [
    ("classic",             True,  "Classical ECDSA-P256  "),
    ("ML-DSA-44",           True,  "Dilithium ML-DSA-44   "),
    ("ML-DSA-65",           True,  "Dilithium ML-DSA-65   "),
    ("ML-DSA-87",           True,  "Dilithium ML-DSA-87   "),
    ("SLH-DSA-shake_128f",  True,  "SPHINCS+ shake_128f   "),
]

print(f"\n{'Backend':<28}  keygen  sign  verify  tamper  wrong_key")
print("-" * 70)

all_pass = True
for backend_id, _, label in BACKENDS:
    try:
        s = make_signer(backend_id)
        pk, sk = s.keygen()
        sig = s.sign(sk, MSG)

        # Correct
        ok_correct  = s.verify(pk, MSG, sig)

        # Tampered message
        ok_tampered = s.verify(pk, MSG_BAD, sig)

        # Wrong key
        pk2, _ = s.keygen()
        ok_wrong_key = s.verify(pk2, MSG, sig)

        correct_pass  = ok_correct  == True
        tamper_pass   = ok_tampered == False
        wrongkey_pass = ok_wrong_key == False

        status = "PASS" if (correct_pass and tamper_pass and wrongkey_pass) else "FAIL"
        if status == "FAIL":
            all_pass = False

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

print()
print("All tests PASSED" if all_pass else "Some tests FAILED")
