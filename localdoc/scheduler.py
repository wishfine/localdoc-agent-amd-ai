"""
异构资源调度器 - Heterogeneous Resource Scheduler

根据任务类型自动选择最优硬件后端：
- document_loading → CPU
- chunking → CPU
- embedding → NPU preferred, GPU second, CPU fallback
- retrieval → GPU preferred, CPU fallback
- generation → GPU preferred, CPU fallback
- report_generation → GPU preferred, CPU fallback
"""

from enum import Enum
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BenchmarkTaskType(Enum):
    """Benchmark task types for heterogeneous scheduling."""
    DOCUMENT_LOADING = "document_loading"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    REPORT_GENERATION = "report_generation"


# Priority rules: task_type -> list of (backend_key, reason)
_TASK_BACKEND_PRIORITY: Dict[BenchmarkTaskType, List[Tuple[str, str]]] = {
    BenchmarkTaskType.DOCUMENT_LOADING: [
        ("cpu", "CPU is optimal for I/O-bound document loading"),
    ],
    BenchmarkTaskType.CHUNKING: [
        ("cpu", "CPU is sufficient for text chunking"),
    ],
    BenchmarkTaskType.EMBEDDING: [
        ("npu", "NPU preferred for embedding with optimized INT8 inference"),
        ("gpu", "GPU fallback for embedding computation"),
        ("cpu", "CPU fallback for embedding"),
    ],
    BenchmarkTaskType.RETRIEVAL: [
        ("gpu", "GPU preferred for parallel vector search"),
        ("cpu", "CPU fallback for retrieval"),
    ],
    BenchmarkTaskType.GENERATION: [
        ("gpu", "GPU preferred for LLM generation"),
        ("cpu", "CPU fallback for generation"),
    ],
    BenchmarkTaskType.REPORT_GENERATION: [
        ("gpu", "GPU preferred for report generation"),
        ("cpu", "CPU fallback for report generation"),
    ],
}


class HeterogeneousScheduler:
    """
    Heterogeneous resource scheduler that maps task types to hardware backends.

    Automatically selects the best available backend for each task type based
    on a configurable priority list.
    """

    def __init__(self, backends: Optional[Dict[str, Any]] = None):
        self.backends: Dict[str, Any] = {}
        self._execution_log: List[Dict[str, Any]] = []
        self._schedule_cache: Dict[str, str] = {}

        if backends is not None:
            self.backends = backends
        else:
            self.auto_detect_backends()

    def auto_detect_backends(self) -> Dict[str, Any]:
        """
        Detect all available hardware backends.

        GPU detection prioritizes ROCm over CUDA:
        - If torch.version.hip is set and torch.cuda.is_available() → ROCmBackend
        - If torch.version.cuda is set and torch.cuda.is_available() → CUDABackend
        """
        detected: Dict[str, Any] = {}

        # Always have CPU available
        from localdoc.backends.cpu_backend import CPUBackend
        detected["cpu"] = CPUBackend()

        # Try GPU (ROCm preferred over CUDA)
        gpu_backend = self._try_detect_gpu()
        if gpu_backend is not None:
            detected["gpu"] = gpu_backend

        # Try NPU (Ryzen AI / ONNX Runtime)
        npu_backend = self._try_detect_npu()
        if npu_backend is not None:
            detected["npu"] = npu_backend

        self.backends = detected
        logger.info("Detected backends: %s", list(detected.keys()))
        return detected

    def _try_detect_gpu(self) -> Optional[Any]:
        """Try to detect a GPU backend. ROCm is checked before CUDA."""
        try:
            import torch
            hip_version = getattr(torch.version, "hip", None)
            cuda_version = getattr(torch.version, "cuda", None)

            if hip_version and torch.cuda.is_available():
                logger.info("Detected AMD ROCm GPU (HIP %s)", hip_version)
                return ROCmBackend()

            if cuda_version and torch.cuda.is_available():
                logger.info("Detected NVIDIA CUDA GPU (CUDA %s)", cuda_version)
                return CUDABackend()
        except ImportError:
            pass

        logger.debug("No GPU backend detected")
        return None

    def _try_detect_npu(self) -> Optional[Any]:
        """Try to detect an AMD NPU backend via ONNX Runtime."""
        try:
            from localdoc.backends.npu_backend import AMDNPUBackend
            backend = AMDNPUBackend()
            if backend.is_available():
                return backend
        except ImportError:
            pass

        logger.debug("No NPU backend detected")
        return None

    def select_backend(self, task_type: BenchmarkTaskType) -> Tuple[Any, str]:
        """
        Select the best available backend for a given task type.

        Returns:
            Tuple of (backend_instance, reason_string).
        """
        if task_type not in _TASK_BACKEND_PRIORITY:
            raise ValueError(f"Unknown task type: {task_type}")

        priority_list = _TASK_BACKEND_PRIORITY[task_type]

        for backend_name, reason in priority_list:
            if backend_name in self.backends:
                # Check if backend is simulated and add warning
                backend = self.backends[backend_name]
                is_simulated = getattr(backend, "name", "").startswith("Simulated")
                if is_simulated:
                    reason = f"{reason} [SIMULATED - not real hardware]"
                logger.info(
                    "Task %s -> backend '%s': %s",
                    task_type.value, backend_name, reason,
                )
                self._schedule_cache[task_type.value] = backend_name
                return backend, reason

        raise RuntimeError(
            f"No backend available for task type '{task_type.value}'. "
            f"Required backends: {[b[0] for b in priority_list]}"
        )

    def execute(
        self,
        task_type: BenchmarkTaskType,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function on the best available backend for the given task type.
        Tracks execution timing and logs the result.
        """
        backend, reason = self.select_backend(task_type)
        backend_name = getattr(backend, "name", type(backend).__name__)

        logger.info(
            "Executing %s on %s (%s)",
            task_type.value, backend_name, reason,
        )

        start_time = time.perf_counter()
        error_occurred = None
        result = None

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error_occurred = str(e)
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            entry = {
                "task_type": task_type.value,
                "backend": backend_name,
                "backend_key": self._find_backend_key(backend),
                "reason": reason,
                "elapsed_seconds": round(elapsed, 6),
                "error": error_occurred,
                "is_simulated": backend_name.startswith("Simulated"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._execution_log.append(entry)
            logger.info(
                "Task %s completed in %.4fs on %s",
                task_type.value, elapsed, backend_name,
            )

        return result

    def _find_backend_key(self, backend: Any) -> str:
        """Find the key name for a given backend instance."""
        for key, instance in self.backends.items():
            if instance is backend:
                return key
        return "unknown"

    def get_schedule_report(self) -> Dict[str, Dict[str, str]]:
        """
        Get a report showing which backend is assigned to each task type.
        Includes simulation warning if the selected backend is simulated.
        """
        report = {}
        available_keys = list(self.backends.keys())

        for task_type in BenchmarkTaskType:
            priority_list = _TASK_BACKEND_PRIORITY.get(task_type, [])
            selected_backend = None
            selected_reason = "No backend available"
            is_simulated = False

            for backend_name, reason in priority_list:
                if backend_name in self.backends:
                    backend = self.backends[backend_name]
                    selected_backend = backend_name
                    selected_reason = reason
                    is_simulated = getattr(backend, "name", "").startswith("Simulated")
                    break

            report[task_type.value] = {
                "backend": selected_backend or "none",
                "reason": selected_reason,
                "available_backends": available_keys,
                "is_simulated": is_simulated,
            }

        return report

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log with timing information."""
        return list(self._execution_log)

    def clear_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()


# ---------------------------------------------------------------------------
# Lightweight backend stubs for auto-detection (used by the scheduler only)
# ---------------------------------------------------------------------------

class ROCmBackend:
    """AMD ROCm GPU backend stub."""
    name = "rocm"
    description = "AMD ROCm GPU backend"

    def is_available(self) -> bool:
        try:
            import torch
            return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        return "ROCmBackend()"


class CUDABackend:
    """NVIDIA CUDA GPU backend stub."""
    name = "cuda"
    description = "NVIDIA CUDA GPU backend"

    def is_available(self) -> bool:
        try:
            import torch
            return bool(getattr(torch.version, "cuda", None)) and torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        return "CUDABackend()"
