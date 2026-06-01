"""
异构资源调度器 - Heterogeneous Resource Scheduler

根据任务类型自动选择最优硬件后端：
- document_loading → CPU
- chunking → CPU
- embedding → NPU preferred, GPU second, CPU fallback
- retrieval → CPU/GPU
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


# Priority rules: task_type -> list of (backend_name, reason)
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
        """
        Initialize the scheduler.

        Args:
            backends: Optional dict mapping backend name -> backend instance.
                      If None, will auto-detect available backends.
        """
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

        Tries to import and instantiate known backend classes. Falls back to
        CPU-only if nothing else is available.

        Returns:
            Dict of backend name -> backend instance.
        """
        detected: Dict[str, Any] = {}

        # Always have CPU available
        detected["cpu"] = self._create_cpu_backend()

        # Try GPU (AMD ROCm or CUDA)
        gpu_backend = self._try_detect_gpu()
        if gpu_backend is not None:
            detected["gpu"] = gpu_backend

        # Try NPU (XDNA / Ryzen AI)
        npu_backend = self._try_detect_npu()
        if npu_backend is not None:
            detected["npu"] = npu_backend

        self.backends = detected
        logger.info("Detected backends: %s", list(detected.keys()))
        return detected

    def _create_cpu_backend(self) -> "CPUBackend":
        """Create a CPU backend instance."""
        return CPUBackend()

    def _try_detect_gpu(self) -> Optional[Any]:
        """Try to detect a GPU backend (CUDA or ROCm)."""
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():
                return CUDABackend()
        except ImportError:
            pass

        try:
            import torch  # noqa: F401
            if hasattr(torch, "hip") and torch.cuda.is_available():
                return ROCmBackend()
        except ImportError:
            pass

        logger.debug("No GPU backend detected")
        return None

    def _try_detect_npu(self) -> Optional[Any]:
        """Try to detect an NPU backend (XDNA / Ryzen AI)."""
        try:
            from .npu_backend import XDNABackend
            backend = XDNABackend()
            if backend.is_available():
                return backend
        except (ImportError, AttributeError):
            pass

        logger.debug("No NPU backend detected")
        return None

    def select_backend(self, task_type: BenchmarkTaskType) -> Tuple[Any, str]:
        """
        Select the best available backend for a given task type.

        Args:
            task_type: The type of task to schedule.

        Returns:
            Tuple of (backend_instance, reason_string).

        Raises:
            RuntimeError: If no backend is available for the task type.
        """
        if task_type not in _TASK_BACKEND_PRIORITY:
            raise ValueError(f"Unknown task type: {task_type}")

        priority_list = _TASK_BACKEND_PRIORITY[task_type]

        for backend_name, reason in priority_list:
            if backend_name in self.backends:
                logger.info(
                    "Task %s -> backend '%s': %s",
                    task_type.value, backend_name, reason,
                )
                self._schedule_cache[task_type.value] = backend_name
                return self.backends[backend_name], reason

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

        Args:
            task_type: The type of task being executed.
            func: The callable to execute.
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            The return value of func.
        """
        backend, reason = self.select_backend(task_type)
        backend_name = type(backend).__name__

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

        Returns:
            Dict mapping task_type -> {backend, reason, available_backends}.
        """
        report = {}
        available_keys = list(self.backends.keys())

        for task_type in BenchmarkTaskType:
            priority_list = _TASK_BACKEND_PRIORITY.get(task_type, [])
            selected_backend = None
            selected_reason = "No backend available"

            for backend_name, reason in priority_list:
                if backend_name in self.backends:
                    selected_backend = backend_name
                    selected_reason = reason
                    break

            report[task_type.value] = {
                "backend": selected_backend or "none",
                "reason": selected_reason,
                "available_backends": available_keys,
            }

        return report

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """
        Get the execution log with timing information.

        Returns:
            List of dicts, each containing task_type, backend, elapsed_seconds, etc.
        """
        return list(self._execution_log)

    def clear_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()


# ---------------------------------------------------------------------------
# Built-in backend stubs
# ---------------------------------------------------------------------------

class CPUBackend:
    """CPU backend - always available."""

    name = "cpu"
    description = "Standard CPU backend"

    def is_available(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "CPUBackend()"


class CUDABackend:
    """NVIDIA CUDA GPU backend."""

    name = "cuda"
    description = "NVIDIA CUDA GPU backend"

    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        return "CUDABackend()"


class ROCmBackend:
    """AMD ROCm GPU backend."""

    name = "rocm"
    description = "AMD ROCm GPU backend"

    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        return "ROCmBackend()"


class SimulatedNPUBackend:
    """Simulated NPU backend for demo / development purposes."""

    name = "simulated_npu"
    description = "Simulated NPU backend (for demo without real XDNA hardware)"

    def __init__(self, simulate_latency_ms: float = 5.0):
        self.simulate_latency_ms = simulate_latency_ms

    def is_available(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"SimulatedNPUBackend(latency={self.simulate_latency_ms}ms)"


class XDNABackend:
    """AMD XDNA NPU backend for Ryzen AI processors."""

    name = "xdna_npu"
    description = "AMD XDNA NPU backend (Ryzen AI)"

    def __init__(self):
        self._available = self._probe()

    def _probe(self) -> bool:
        """Probe for XDNA device availability."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return "VitisAIExecutionProvider" in providers
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def __repr__(self) -> str:
        return f"XDNABackend(available={self._available})"
