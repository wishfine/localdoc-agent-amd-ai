"""ROCm runtime safety probes.

PyTorch can report ``torch.version.hip`` and ``torch.cuda.is_available()`` as
true while the first real tensor operation still aborts the process in a native
ROCm library. Run the risky tensor operation in a child process so the main
experiment runner can mark ROCm as unavailable instead of crashing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from typing import Tuple


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_gpu_backend_disabled() -> bool:
    """Return true when tests or users explicitly disable real GPU execution."""
    return _truthy_env("LOCALDOC_DISABLE_GPU_BACKEND")


@lru_cache(maxsize=1)
def rocm_tensor_probe(timeout_s: int = 20) -> Tuple[bool, str]:
    """Validate that a minimal ROCm tensor operation works in a subprocess."""
    if is_gpu_backend_disabled():
        return False, "disabled by LOCALDOC_DISABLE_GPU_BACKEND"
    if _truthy_env("LOCALDOC_SKIP_ROCM_TENSOR_PROBE"):
        return True, "skipped by LOCALDOC_SKIP_ROCM_TENSOR_PROBE"

    code = r"""
import torch

hip = getattr(torch.version, "hip", None)
if not hip:
    raise SystemExit("torch.version.hip is empty")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
y = (x * x).sum()
torch.cuda.synchronize()
print("ok", float(y.cpu()))
"""
    env = os.environ.copy()
    env["LOCALDOC_SKIP_ROCM_TENSOR_PROBE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"tensor probe timeout after {timeout_s}s"
    except Exception as exc:  # pragma: no cover - defensive path
        return False, f"tensor probe failed: {type(exc).__name__}: {exc}"

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, f"tensor probe exit_code={proc.returncode}; {output}"
    return True, output or "tensor probe ok"
