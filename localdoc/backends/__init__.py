from .cpu_backend import CPUBackend
from .gpu_backend import AMDGPUBackend
from .npu_backend import AMDNPUBackend
from .simulated_npu import SimulatedNPUBackend

# Optional: Local LLM backend (requires torch + transformers)
try:
    from .local_llm_backend import LocalLLMBackend
except Exception:
    LocalLLMBackend = None

__all__ = [
    "CPUBackend",
    "AMDGPUBackend",
    "AMDNPUBackend",
    "SimulatedNPUBackend",
    "LocalLLMBackend",
]
