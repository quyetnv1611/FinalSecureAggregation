from __future__ import annotations

"""Probe which crypto backend modules are importable in the current runtime.

Use this in Colab before running strict CUDA benchmarks.

Examples:
  python check_cuda_crypto_modules.py
  python check_cuda_crypto_modules.py --include-skeleton
"""

import argparse
import importlib
import importlib.util
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbeResult:
    name: str
    importable: bool
    module_path: str | None
    details: str


def _probe_module(name: str) -> ProbeResult:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return ProbeResult(name=name, importable=False, module_path=None, details="not importable")

    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return ProbeResult(name=name, importable=False, module_path=getattr(spec, "origin", None), details=f"import failed: {exc}")

    attrs = []
    for attr_name in ("KeyEncapsulation", "Signature", "ML_KEM_512", "ML_KEM_768", "ML_DSA_65", "ML_DSA_87"):
        if hasattr(module, attr_name):
            attrs.append(attr_name)
    if hasattr(module, "__all__"):
        attrs.append("__all__")
    details = "attrs=" + (",".join(attrs) if attrs else "none")
    return ProbeResult(name=name, importable=True, module_path=getattr(spec, "origin", None), details=details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect importable crypto backend modules")
    parser.add_argument("--include-skeleton", action="store_true", help="Include secagg.cuda_adapter_skeleton in the probe list")
    args = parser.parse_args()

    print(f"[probe] torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[probe] torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")

    candidates = [
        "pqc_cuda_kyber",
        "kyber_cuda",
        "liboqs_cuda",
        "cuDilithium",
        "dilithium_cuda",
        "oqs",
        "liboqs",
        "kyber_py",
        "dilithium_py",
        "slhdsa",
    ]
    if args.include_skeleton:
        candidates.append("secagg.cuda_adapter_skeleton")

    results = [_probe_module(name) for name in candidates]

    print("[probe] importable modules:")
    for result in results:
        status = "YES" if result.importable else "no"
        print(f"  - {result.name:<28} {status:<3} {result.details} path={result.module_path}")

    print("[probe] recommended mappings:")
    if any(r.name in {"pqc_cuda_kyber", "kyber_cuda", "liboqs_cuda"} and r.importable for r in results):
        print("  * KEM: use the first importable CUDA module among pqc_cuda_kyber / kyber_cuda / liboqs_cuda")
    if any(r.name in {"cuDilithium", "dilithium_cuda", "liboqs_cuda"} and r.importable for r in results):
        print("  * SIG: use the first importable CUDA module among cuDilithium / dilithium_cuda / liboqs_cuda")
    if any(r.name == "oqs" and r.importable for r in results):
        print("  * CPU fallback: oqs (liboqs-python) is available")
    if any(r.name == "liboqs" and r.importable for r in results):
        print("  * CPU fallback: liboqs module is available")


if __name__ == "__main__":
    main()
