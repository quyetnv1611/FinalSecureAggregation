

import sys
import os

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from secagg.crypto_mlkem import SecAggregatorMLKEM




def _abs_max(a: "np.ndarray", b: "np.ndarray") -> float:
    return float(np.max(np.abs(a - b)))




def test_pairwise_mask_cancellation():
    print("=== Test ML-KEM-768 — pairwise mask cancellation ===")
    u = SecAggregatorMLKEM(shape=(4, 4))
    v = SecAggregatorMLKEM(shape=(4, 4))
    sid_u, sid_v = "AAAA", "BBBB"

    peer_eks = {sid_u: u.public_key, sid_v: v.public_key}

    ct_from_u = u.generate_ciphertexts(peer_eks, sid_u)
    v.generate_ciphertexts(peer_eks, sid_v)
    v.receive_ciphertexts({sid_u: ct_from_u[sid_v]})

    u.set_weights(np.ones((4, 4), dtype="float64"))
    v.set_weights(np.ones((4, 4), dtype="float64"))
    masked_u = u.prepare_masked_gradient()
    masked_v = v.prepare_masked_gradient()

    server_sum = masked_u + masked_v
    expected = np.ones((4, 4)) * 2 + (-u.private_mask()) + (-v.private_mask())
    err = _abs_max(server_sum, expected)
    print(f"  Mask cancellation error: {err:.8f}  (should be 0.0)")
    assert err < 1e-5, f"FAILED: error={err}"
    print("  PASSED")


def test_all_security_levels():
    print("\n=== All 3 security levels ===")
    for level in ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]:
        a = SecAggregatorMLKEM(shape=(3, 3), security_level=level)
        b = SecAggregatorMLKEM(shape=(3, 3), security_level=level)
        sa, sb = "A001", "B002"
        pks = {sa: a.public_key, sb: b.public_key}
        ct_a = a.generate_ciphertexts(pks, sa)
        b.generate_ciphertexts(pks, sb)
        b.receive_ciphertexts({sa: ct_a[sb]})
        a.set_weights(np.eye(3, dtype="float64"))
        b.set_weights(np.eye(3, dtype="float64"))
        s = a.prepare_masked_gradient() + b.prepare_masked_gradient()
        e = np.eye(3) * 2 + (-a.private_mask()) + (-b.private_mask())
        err = _abs_max(s, e)
        ok = err < 1e-5
        status = "PASSED" if ok else "FAILED"
        print(
            f"  {level:<15s}  ek={a.encapsulation_key_size}B"
            f"  ct={a.ciphertext_size}B  => {status}"
        )
        assert ok, f"{level} FAILED: error={err}"


def test_dropout_correction():
    print("\n=== Reveal correction for 1 dropout ===")
    a = SecAggregatorMLKEM(shape=(3, 3))
    b = SecAggregatorMLKEM(shape=(3, 3))
    c = SecAggregatorMLKEM(shape=(3, 3))
    sa, sb, sc = "A001", "B002", "C003"
    pks = {sa: a.public_key, sb: b.public_key, sc: c.public_key}

    ct_a = a.generate_ciphertexts(pks, sa)
    ct_b = b.generate_ciphertexts(pks, sb)
    c.generate_ciphertexts(pks, sc)
    b.receive_ciphertexts({sa: ct_a[sb]})
    c.receive_ciphertexts({sa: ct_a[sc], sb: ct_b[sc]})

    a.set_weights(np.ones((3, 3), dtype="float64") * 1.0)
    b.set_weights(np.ones((3, 3), dtype="float64") * 2.0)
    # c will be a dropout — never uploads

    masked_a = a.prepare_masked_gradient()
    masked_b = b.prepare_masked_gradient()

    corr_a = a.reveal_pairwise_masks([sc])
    corr_b = b.reveal_pairwise_masks([sc])

    server_agg = (masked_a + masked_b + corr_a + corr_b
                  + a.private_mask() + b.private_mask())
    expected_agg = np.ones((3, 3)) * 1.0 + np.ones((3, 3)) * 2.0
    err = _abs_max(server_agg, expected_agg)
    print(f"  Dropout correction error: {err:.8f}  (should be 0.0)")
    assert err < 1e-4, f"FAILED: error={err}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_pairwise_mask_cancellation()
    test_all_security_levels()
    test_dropout_correction()
    print("\n=== All tests PASSED ===")
