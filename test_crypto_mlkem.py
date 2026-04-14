import sys
sys.path.insert(0, r'C:\Users\quyet\OneDrive\Desktop\SecureAggregation-master\SecureAggregation-master')
from secagg.crypto_mlkem import SecAggregatorMLKEM
import numpy as np

print('=== Test ML-KEM-768 — pairwise mask cancellation ===')
u = SecAggregatorMLKEM(shape=(4,4))
v = SecAggregatorMLKEM(shape=(4,4))
sid_u, sid_v = 'AAAA', 'BBBB'

peer_eks = {sid_u: u.public_key, sid_v: v.public_key}

# Round 0.5: u encaps to v; server routes ct to v keyed by sender=sid_u
ct_from_u = u.generate_ciphertexts(peer_eks, sid_u)
v.generate_ciphertexts(peer_eks, sid_v)  # nothing to encaps (no higher SID)
v.receive_ciphertexts({sid_u: ct_from_u[sid_v]})

# Round 2: masked gradients
u.set_weights(np.ones((4,4), dtype='float64'))
v.set_weights(np.ones((4,4), dtype='float64'))
masked_u = u.prepare_masked_gradient()
masked_v = v.prepare_masked_gradient()

# Server sums: pairwise masks cancel, only private masks remain
server_sum = masked_u + masked_v
expected   = np.ones((4,4)) * 2 + (-u.private_mask()) + (-v.private_mask())
diff = float(np.max(np.abs(server_sum - expected)))
print(f'Mask cancellation error: {diff:.8f}  (should be 0.0)')
print('PASSED' if diff < 1e-5 else 'FAILED')

print()
print('=== All 3 security levels ===')
for level in ['ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024']:
    a = SecAggregatorMLKEM(shape=(3,3), security_level=level)
    b = SecAggregatorMLKEM(shape=(3,3), security_level=level)
    sa, sb = 'A001', 'B002'
    pks = {sa: a.public_key, sb: b.public_key}
    ct_a = a.generate_ciphertexts(pks, sa)
    b.generate_ciphertexts(pks, sb)
    b.receive_ciphertexts({sa: ct_a[sb]})
    a.set_weights(np.eye(3, dtype='float64'))
    b.set_weights(np.eye(3, dtype='float64'))
    s = a.prepare_masked_gradient() + b.prepare_masked_gradient()
    e = np.eye(3)*2 + (-a.private_mask()) + (-b.private_mask())
    ok = float(np.max(np.abs(s - e))) < 1e-5
    status = 'PASSED' if ok else 'FAILED'
    print(f'  {level:<15s}  ek={a.encapsulation_key_size}B  ct={a.ciphertext_size}B  => {status}')

print()
print('=== Reveal correction for 1 dropout ===')
a = SecAggregatorMLKEM(shape=(3,3))
b = SecAggregatorMLKEM(shape=(3,3))
c = SecAggregatorMLKEM(shape=(3,3))
sa, sb, sc = 'A001', 'B002', 'C003'
pks = {sa: a.public_key, sb: b.public_key, sc: c.public_key}

# Ciphertext exchange: a encaps to b and c; b encaps to c
ct_a = a.generate_ciphertexts(pks, sa)
ct_b = b.generate_ciphertexts(pks, sb)
c.generate_ciphertexts(pks, sc)
b.receive_ciphertexts({sa: ct_a[sb]})
c.receive_ciphertexts({sa: ct_a[sc], sb: ct_b[sc]})

a.set_weights(np.ones((3,3), dtype='float64') * 1.0)
b.set_weights(np.ones((3,3), dtype='float64') * 2.0)
# c will be a dropout

masked_a = a.prepare_masked_gradient()
masked_b = b.prepare_masked_gradient()
# c never uploads

# Server asks a and b to reveal correction for dropout c
corr_a = a.reveal_pairwise_masks([sc])
corr_b = b.reveal_pairwise_masks([sc])

# Server reconstructs: sum of surviving masks + corrections = true sum of survivors
server_agg = masked_a + masked_b + corr_a + corr_b + a.private_mask() + b.private_mask()
expected_agg = np.ones((3,3)) * 1.0 + np.ones((3,3)) * 2.0
diff_dropout = float(np.max(np.abs(server_agg - expected_agg)))
print(f'Dropout correction error: {diff_dropout:.8f}  (should be 0.0)')
print('PASSED' if diff_dropout < 1e-4 else 'FAILED')
