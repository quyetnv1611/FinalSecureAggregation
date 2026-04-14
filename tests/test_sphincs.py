
import slhdsa
import slhdsa.lowlevel.slhdsa as _ll
from slhdsa.slhdsa import SecretKey

msg = b"test-message"
variants = [
	slhdsa.shake_128f,
	slhdsa.shake_128s,
	slhdsa.shake_192f,
	slhdsa.shake_192s,
	slhdsa.shake_256f,
	slhdsa.shake_256s,
]

for variant in variants:
	sk_tuple, pk_tuple = _ll.keygen(variant)
	sk_obj = SecretKey(sk_tuple, variant)
	pk_obj = sk_obj.pubkey
	sig = sk_obj.sign_pure(msg)
	ok = pk_obj.verify_pure(msg, sig)
	print(f"Variant: {variant}, Verify result: {ok}")