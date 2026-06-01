"""Tests for HeterogeneousScheduler class."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from localdoc.scheduler import HeterogeneousScheduler, BenchmarkTaskType
from localdoc.backends.cpu_backend import CPUBackend


# ---------------------------------------------------------------------------
# Helper: create a fake backend with a given name
# ---------------------------------------------------------------------------

class _FakeBackend:
    """Minimal backend stub for testing."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def __repr__(self):
        return f"FakeBackend({self._name})"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cpu_backend_selection():
    """Test CPU selected for document_loading and chunking."""
    scheduler = HeterogeneousScheduler(backends={"cpu": CPUBackend()})

    backend_load, _ = scheduler.select_backend(BenchmarkTaskType.DOCUMENT_LOADING)
    backend_chunk, _ = scheduler.select_backend(BenchmarkTaskType.CHUNKING)

    assert isinstance(backend_load, CPUBackend)
    assert isinstance(backend_chunk, CPUBackend)


def test_embedding_priority():
    """Test NPU > GPU > CPU for embedding."""
    # With all backends
    scheduler_all = HeterogeneousScheduler(
        backends={
            "cpu": CPUBackend(),
            "gpu": _FakeBackend("GPU"),
            "npu": _FakeBackend("NPU"),
        }
    )
    backend, _ = scheduler_all.select_backend(BenchmarkTaskType.EMBEDDING)
    assert backend.name == "NPU"

    # With only GPU and CPU
    scheduler_gpu = HeterogeneousScheduler(
        backends={
            "cpu": CPUBackend(),
            "gpu": _FakeBackend("GPU"),
        }
    )
    backend, _ = scheduler_gpu.select_backend(BenchmarkTaskType.EMBEDDING)
    assert backend.name == "GPU"

    # With only CPU
    scheduler_cpu = HeterogeneousScheduler(backends={"cpu": CPUBackend()})
    backend, _ = scheduler_cpu.select_backend(BenchmarkTaskType.EMBEDDING)
    assert isinstance(backend, CPUBackend)


def test_generation_priority():
    """Test GPU > CPU for generation."""
    scheduler_gpu = HeterogeneousScheduler(
        backends={
            "cpu": CPUBackend(),
            "gpu": _FakeBackend("GPU"),
        }
    )
    backend, _ = scheduler_gpu.select_backend(BenchmarkTaskType.GENERATION)
    assert backend.name == "GPU"

    scheduler_cpu = HeterogeneousScheduler(backends={"cpu": CPUBackend()})
    backend, _ = scheduler_cpu.select_backend(BenchmarkTaskType.GENERATION)
    assert isinstance(backend, CPUBackend)


def test_fallback_to_cpu():
    """Test falls back to CPU when others unavailable."""
    scheduler = HeterogeneousScheduler(backends={"cpu": CPUBackend()})

    for task_type in BenchmarkTaskType:
        backend, _ = scheduler.select_backend(task_type)
        assert isinstance(backend, CPUBackend)


def test_execution_tracking():
    """Test executions are logged."""
    scheduler = HeterogeneousScheduler(backends={"cpu": CPUBackend()})

    scheduler.execute(BenchmarkTaskType.EMBEDDING, lambda: None)
    scheduler.execute(BenchmarkTaskType.GENERATION, lambda: None)

    log = scheduler.get_execution_log()
    assert len(log) == 2
    assert all("task_type" in e for e in log)
    assert all("backend" in e for e in log)
    assert all("elapsed_seconds" in e for e in log)


def test_schedule_report():
    """Test report format is correct."""
    scheduler = HeterogeneousScheduler(
        backends={
            "cpu": CPUBackend(),
            "gpu": _FakeBackend("GPU"),
            "npu": _FakeBackend("NPU"),
        }
    )

    report = scheduler.get_schedule_report()

    # Should have an entry for each task type
    for task_type in BenchmarkTaskType:
        assert task_type.value in report
        entry = report[task_type.value]
        assert "backend" in entry
        assert "reason" in entry
        assert "available_backends" in entry

    # Verify specific scheduling rules
    assert report["document_loading"]["backend"] == "cpu"
    assert report["chunking"]["backend"] == "cpu"
    assert report["embedding"]["backend"] == "npu"
    assert report["generation"]["backend"] == "gpu"


def test_auto_detect():
    """Test auto detection works without crashing."""
    scheduler = HeterogeneousScheduler()

    # Should have at least CPU backend
    assert "cpu" in scheduler.backends
    assert scheduler.backends["cpu"].is_available()


def test_execute_returns_value():
    """Test that execute() returns the function's return value."""
    scheduler = HeterogeneousScheduler(backends={"cpu": CPUBackend()})

    result = scheduler.execute(
        BenchmarkTaskType.CHUNKING, lambda: 42
    )
    assert result == 42


def test_execute_timing():
    """Test that execute() records timing."""
    import time

    scheduler = HeterogeneousScheduler(backends={"cpu": CPUBackend()})

    scheduler.execute(
        BenchmarkTaskType.CHUNKING,
        lambda: time.sleep(0.01),
    )

    log = scheduler.get_execution_log()
    assert len(log) == 1
    assert log[0]["elapsed_seconds"] >= 0.005  # at least ~10ms


def test_clear_log():
    """Test clearing the execution log."""
    scheduler = HeterogeneousScheduler(backends={"cpu": CPUBackend()})

    scheduler.execute(BenchmarkTaskType.CHUNKING, lambda: None)
    assert len(scheduler.get_execution_log()) == 1

    scheduler.clear_log()
    assert len(scheduler.get_execution_log()) == 0
