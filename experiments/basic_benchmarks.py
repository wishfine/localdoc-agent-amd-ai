"""
Basic heterogeneous-computing benchmarks for the course grading rubric.

Outputs:
- results/matmul_benchmark.csv
- results/precision_compare.csv
- results/mlp_train_log.csv

The script always runs a real CPU baseline with NumPy. If PyTorch with ROCm/HIP
is available, it also runs the same workloads on the AMD GPU through the PyTorch
``cuda`` device API used by ROCm.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from localdoc.backends.rocm_safety import rocm_tensor_probe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _rocm_available(torch_mod) -> bool:
    if torch_mod is None:
        return False
    hip_version = getattr(torch_mod.version, "hip", None)
    if not (hip_version and torch_mod.cuda.is_available()):
        return False
    probe_ok, _ = rocm_tensor_probe()
    return probe_ok


def _rocm_note(torch_mod) -> str:
    if torch_mod is None:
        return "PyTorch not installed; ROCm GPU benchmark unavailable"
    hip_version = getattr(torch_mod.version, "hip", None)
    if not hip_version:
        return "PyTorch installed but torch.version.hip is empty; not a ROCm build"
    if not torch_mod.cuda.is_available():
        return "ROCm PyTorch detected but torch.cuda.is_available() is False"
    probe_ok, probe_note = rocm_tensor_probe()
    if not probe_ok:
        return f"ROCm PyTorch detected but tensor probe failed; GPU benchmark disabled. {probe_note}"
    return "ROCm GPU available"


def _matmul_gflops(size: int, avg_ms: float) -> float:
    if avg_ms <= 0:
        return 0.0
    return round((2.0 * size * size * size) / (avg_ms / 1000.0) / 1e9, 4)


def _sync_device(torch_mod, device: str) -> Optional[Callable[[], None]]:
    if device == "cuda":
        return torch_mod.cuda.synchronize
    return None


def _device_name(torch_mod, device: str) -> str:
    if device == "cuda":
        return torch_mod.cuda.get_device_name(0)
    return "torch_cpu"


def _write_csv(rows: List[Dict[str, Any]], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in rows])
    print(f"  [saved] {path}")
    return path


def _time_repeated(
    fn: Callable[[], Any],
    repeats: int,
    sync: Optional[Callable[[], None]] = None,
    warmup: int = 1,
) -> Tuple[float, float, float, float]:
    for _ in range(warmup):
        fn()
        if sync:
            sync()

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)

    avg = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return avg, std, min(times), max(times)


def benchmark_matmul(sizes: List[int], repeats: int, seed: int = 42) -> List[Dict[str, Any]]:
    print("\n[1/3] Matrix multiplication benchmark ...")
    rng = np.random.default_rng(seed)
    torch_mod = _import_torch()
    rocm_ok = _rocm_available(torch_mod)
    rows: List[Dict[str, Any]] = []
    cpu_avg_by_size: Dict[int, float] = {}
    torch_cpu_avg_by_size: Dict[int, float] = {}

    for size in sizes:
        a = rng.standard_normal((size, size)).astype(np.float32)
        b = rng.standard_normal((size, size)).astype(np.float32)

        holder: Dict[str, Any] = {}

        def run_cpu():
            holder["cpu"] = a @ b

        avg, std, min_ms, max_ms = _time_repeated(run_cpu, repeats)
        cpu_avg_by_size[size] = avg
        rows.append({
            "experiment": "matmul",
            "backend": "CPU",
            "device": "numpy_cpu",
            "backend_library": "numpy",
            "available": True,
            "measurement_type": "real_cpu",
            "size": size,
            "dtype": "fp32",
            "repeats": repeats,
            "avg_ms": round(avg, 4),
            "std_ms": round(std, 4),
            "min_ms": round(min_ms, 4),
            "max_ms": round(max_ms, 4),
            "gflops": _matmul_gflops(size, avg),
            "speedup_vs_cpu": 1.0,
            "speedup_baseline_backend": "CPU",
            "torch_version": getattr(torch_mod, "__version__", ""),
            "torch_hip_version": getattr(getattr(torch_mod, "version", None), "hip", "") if torch_mod else "",
            "note": "Real CPU NumPy matmul",
        })
        print(f"  [matmul] CPU size={size:<5} avg={avg:>8.3f} ms")

        if torch_mod is not None:
            ta = torch_mod.tensor(a, dtype=torch_mod.float32, device="cpu")
            tb = torch_mod.tensor(b, dtype=torch_mod.float32, device="cpu")

            def run_torch_cpu():
                holder["torch_cpu"] = ta @ tb

            avg, std, min_ms, max_ms = _time_repeated(run_torch_cpu, repeats)
            torch_cpu_avg_by_size[size] = avg
            rows.append({
                "experiment": "matmul",
                "backend": "Torch_CPU",
                "device": "torch_cpu",
                "backend_library": "pytorch",
                "available": True,
                "measurement_type": "real_torch_cpu",
                "size": size,
                "dtype": "fp32",
                "repeats": repeats,
                "avg_ms": round(avg, 4),
                "std_ms": round(std, 4),
                "min_ms": round(min_ms, 4),
                "max_ms": round(max_ms, 4),
                "gflops": _matmul_gflops(size, avg),
                "speedup_vs_cpu": round(cpu_avg_by_size[size] / avg, 4) if avg > 0 else "",
                "speedup_baseline_backend": "CPU",
                "torch_version": torch_mod.__version__,
                "torch_hip_version": getattr(torch_mod.version, "hip", ""),
                "note": "Real CPU PyTorch matmul",
            })
            print(f"  [matmul] Torch_CPU size={size:<5} avg={avg:>8.3f} ms")

        if not rocm_ok:
            rows.append({
                "experiment": "matmul",
                "backend": "ROCm_GPU",
                "device": "cuda",
                "backend_library": "pytorch",
                "available": False,
                "measurement_type": "unavailable",
                "size": size,
                "dtype": "fp32",
                "repeats": repeats,
                "note": _rocm_note(torch_mod),
            })
            continue

        assert torch_mod is not None
        torch_mod.manual_seed(seed + size)
        gpu_a = torch_mod.randn((size, size), device="cuda", dtype=torch_mod.float32)
        gpu_b = torch_mod.randn((size, size), device="cuda", dtype=torch_mod.float32)

        def run_gpu():
            holder["gpu"] = gpu_a @ gpu_b

        avg, std, min_ms, max_ms = _time_repeated(run_gpu, repeats, sync=torch_mod.cuda.synchronize)
        baseline_backend = "Torch_CPU" if size in torch_cpu_avg_by_size else "CPU"
        baseline_avg = torch_cpu_avg_by_size.get(size, cpu_avg_by_size[size])
        rows.append({
            "experiment": "matmul",
            "backend": "ROCm_GPU",
            "device": torch_mod.cuda.get_device_name(0),
            "backend_library": "pytorch",
            "available": True,
            "measurement_type": "real_rocm_gpu",
            "size": size,
            "dtype": "fp32",
            "repeats": repeats,
            "avg_ms": round(avg, 4),
            "std_ms": round(std, 4),
            "min_ms": round(min_ms, 4),
            "max_ms": round(max_ms, 4),
            "gflops": _matmul_gflops(size, avg),
            "speedup_vs_cpu": round(baseline_avg / avg, 4) if avg > 0 else "",
            "speedup_baseline_backend": baseline_backend,
            "torch_version": torch_mod.__version__,
            "torch_hip_version": getattr(torch_mod.version, "hip", ""),
            "note": "Real ROCm GPU matmul via PyTorch HIP",
        })
        print(f"  [matmul] ROCm_GPU size={size:<5} avg={avg:>8.3f} ms")

    return rows


def benchmark_precision(sizes: List[int], repeats: int, seed: int = 123) -> List[Dict[str, Any]]:
    print("\n[2/3] FP32/FP16 precision comparison ...")
    rng = np.random.default_rng(seed)
    torch_mod = _import_torch()
    rocm_ok = _rocm_available(torch_mod)
    rows: List[Dict[str, Any]] = []
    torch_cpu_fp32_by_size: Dict[int, float] = {}

    for size in sizes:
        a32 = rng.standard_normal((size, size)).astype(np.float32)
        b32 = rng.standard_normal((size, size)).astype(np.float32)
        a16 = a32.astype(np.float16)
        b16 = b32.astype(np.float16)
        holder: Dict[str, Any] = {}

        def run_fp32_cpu():
            holder["fp32"] = a32 @ b32

        def run_fp16_cpu():
            holder["fp16"] = a16 @ b16

        fp32_ms, _, _, _ = _time_repeated(run_fp32_cpu, repeats)
        fp16_ms, _, _, _ = _time_repeated(run_fp16_cpu, repeats)
        ref = a32 @ b32
        approx = (a16 @ b16).astype(np.float32)
        err = np.abs(ref - approx)
        rel_l2 = float(np.linalg.norm(ref - approx) / max(np.linalg.norm(ref), 1e-12))
        rows.append({
            "experiment": "precision_compare",
            "backend": "CPU",
            "device": "numpy_cpu",
            "backend_library": "numpy",
            "available": True,
            "measurement_type": "real_cpu",
            "size": size,
            "repeats": repeats,
            "fp32_ms": round(fp32_ms, 4),
            "fp16_ms": round(fp16_ms, 4),
            "speedup_fp16_vs_fp32": round(fp32_ms / fp16_ms, 4) if fp16_ms > 0 else "",
            "max_abs_error": round(float(err.max()), 8),
            "mean_abs_error": round(float(err.mean()), 8),
            "relative_l2_error": round(rel_l2, 8),
            "note": "Real CPU NumPy precision comparison",
        })
        print(f"  [precision] CPU size={size:<5} fp32={fp32_ms:>8.3f} ms fp16={fp16_ms:>8.3f} ms")

        if torch_mod is not None:
            ta32 = torch_mod.tensor(a32, device="cpu", dtype=torch_mod.float32)
            tb32 = torch_mod.tensor(b32, device="cpu", dtype=torch_mod.float32)
            ta16 = ta32.to(torch_mod.float16)
            tb16 = tb32.to(torch_mod.float16)

            def run_fp32_torch_cpu():
                holder["torch_cpu_fp32"] = ta32 @ tb32

            def run_fp16_torch_cpu():
                holder["torch_cpu_fp16"] = ta16 @ tb16

            fp32_torch_ms, _, _, _ = _time_repeated(run_fp32_torch_cpu, repeats)
            fp16_torch_ms, _, _, _ = _time_repeated(run_fp16_torch_cpu, repeats)
            torch_cpu_fp32_by_size[size] = fp32_torch_ms
            ref_t = (ta32 @ tb32).float()
            approx_t = (ta16 @ tb16).float()
            err_t = (ref_t - approx_t).abs()
            rel_l2_t = torch_mod.linalg.vector_norm(ref_t - approx_t) / torch_mod.linalg.vector_norm(ref_t).clamp_min(1e-12)
            rows.append({
                "experiment": "precision_compare",
                "backend": "Torch_CPU",
                "device": "torch_cpu",
                "backend_library": "pytorch",
                "available": True,
                "measurement_type": "real_torch_cpu",
                "size": size,
                "repeats": repeats,
                "fp32_ms": round(fp32_torch_ms, 4),
                "fp16_ms": round(fp16_torch_ms, 4),
                "speedup_fp16_vs_fp32": round(fp32_torch_ms / fp16_torch_ms, 4) if fp16_torch_ms > 0 else "",
                "max_abs_error": round(float(err_t.max().item()), 8),
                "mean_abs_error": round(float(err_t.mean().item()), 8),
                "relative_l2_error": round(float(rel_l2_t.item()), 8),
                "torch_version": torch_mod.__version__,
                "torch_hip_version": getattr(torch_mod.version, "hip", ""),
                "note": "Real CPU PyTorch FP32/FP16 comparison",
            })
            print(f"  [precision] Torch_CPU size={size:<5} fp32={fp32_torch_ms:>8.3f} ms fp16={fp16_torch_ms:>8.3f} ms")

        if not rocm_ok:
            rows.append({
                "experiment": "precision_compare",
                "backend": "ROCm_GPU",
                "device": "cuda",
                "backend_library": "pytorch",
                "available": False,
                "measurement_type": "unavailable",
                "size": size,
                "repeats": repeats,
                "note": _rocm_note(torch_mod),
            })
            continue

        assert torch_mod is not None
        torch_mod.manual_seed(seed + size)
        gpu_a32 = torch_mod.randn((size, size), device="cuda", dtype=torch_mod.float32)
        gpu_b32 = torch_mod.randn((size, size), device="cuda", dtype=torch_mod.float32)
        gpu_a16 = gpu_a32.to(torch_mod.float16)
        gpu_b16 = gpu_b32.to(torch_mod.float16)

        def run_fp32_gpu():
            holder["gpu_fp32"] = gpu_a32 @ gpu_b32

        def run_fp16_gpu():
            holder["gpu_fp16"] = gpu_a16 @ gpu_b16

        fp32_ms, _, _, _ = _time_repeated(run_fp32_gpu, repeats, sync=torch_mod.cuda.synchronize)
        fp16_ms, _, _, _ = _time_repeated(run_fp16_gpu, repeats, sync=torch_mod.cuda.synchronize)
        ref_t = (gpu_a32 @ gpu_b32).float()
        approx_t = (gpu_a16 @ gpu_b16).float()
        err_t = (ref_t - approx_t).abs()
        rel_l2_t = torch_mod.linalg.vector_norm(ref_t - approx_t) / torch_mod.linalg.vector_norm(ref_t).clamp_min(1e-12)
        rows.append({
            "experiment": "precision_compare",
            "backend": "ROCm_GPU",
            "device": torch_mod.cuda.get_device_name(0),
            "backend_library": "pytorch",
            "available": True,
            "measurement_type": "real_rocm_gpu",
            "size": size,
            "repeats": repeats,
            "fp32_ms": round(fp32_ms, 4),
            "fp16_ms": round(fp16_ms, 4),
            "speedup_fp16_vs_fp32": round(fp32_ms / fp16_ms, 4) if fp16_ms > 0 else "",
            "speedup_fp32_vs_torch_cpu": round(torch_cpu_fp32_by_size[size] / fp32_ms, 4)
            if fp32_ms > 0 and size in torch_cpu_fp32_by_size else "",
            "max_abs_error": round(float(err_t.max().item()), 8),
            "mean_abs_error": round(float(err_t.mean().item()), 8),
            "relative_l2_error": round(float(rel_l2_t.item()), 8),
            "torch_version": torch_mod.__version__,
            "torch_hip_version": getattr(torch_mod.version, "hip", ""),
            "note": "Real ROCm GPU FP32/FP16 comparison via PyTorch HIP",
        })
        print(f"  [precision] ROCm_GPU size={size:<5} fp32={fp32_ms:>8.3f} ms fp16={fp16_ms:>8.3f} ms")

    return rows


def _make_classification(samples: int, input_dim: int, classes: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((samples, input_dim)).astype(np.float32)
    true_w = rng.standard_normal((input_dim, classes)).astype(np.float32)
    logits = x @ true_w + 0.1 * rng.standard_normal((samples, classes)).astype(np.float32)
    y = logits.argmax(axis=1).astype(np.int64)
    return x, y


def _numpy_mlp_train(
    epochs: int,
    samples: int,
    batch_size: int,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    x, y = _make_classification(samples=samples, input_dim=32, classes=3, seed=seed)
    rng = np.random.default_rng(seed + 1)
    w1 = (rng.standard_normal((32, 64)) * 0.05).astype(np.float32)
    b1 = np.zeros((64,), dtype=np.float32)
    w2 = (rng.standard_normal((64, 3)) * 0.05).astype(np.float32)
    b2 = np.zeros((3,), dtype=np.float32)
    lr = 0.08
    rows: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(samples)
        t0 = time.perf_counter()
        for start in range(0, samples, batch_size):
            idx = order[start:start + batch_size]
            xb = x[idx]
            yb = y[idx]
            n = len(idx)

            z1 = xb @ w1 + b1
            h1 = np.maximum(z1, 0.0)
            logits = h1 @ w2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            dlogits = probs
            dlogits[np.arange(n), yb] -= 1.0
            dlogits /= n
            dw2 = h1.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh1 = dlogits @ w2.T
            dz1 = dh1 * (z1 > 0)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)

            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1

        epoch_ms = (time.perf_counter() - t0) * 1000
        z1 = x @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(samples), y] + 1e-12).mean()
        pred = probs.argmax(axis=1)
        acc = (pred == y).mean()
        rows.append({
            "experiment": "mlp_train",
            "backend": "CPU",
            "device": "numpy_cpu",
            "backend_library": "numpy",
            "available": True,
            "measurement_type": "real_cpu",
            "dtype": "fp32",
            "epoch": epoch,
            "samples": samples,
            "batch_size": batch_size,
            "loss": round(float(loss), 6),
            "accuracy": round(float(acc), 6),
            "epoch_time_ms": round(epoch_ms, 4),
            "samples_per_second": round(samples / (epoch_ms / 1000.0), 2) if epoch_ms > 0 else "",
            "note": "Real CPU NumPy MLP training with forward/backward/update",
        })
    return rows


def _torch_mlp_train(
    torch_mod,
    epochs: int,
    samples: int,
    batch_size: int,
    device: str,
    backend: str,
    measurement_type: str,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    torch_mod.manual_seed(seed)
    x_np, y_np = _make_classification(samples=samples, input_dim=32, classes=3, seed=seed)
    x = torch_mod.tensor(x_np, device=device, dtype=torch_mod.float32)
    y = torch_mod.tensor(y_np, device=device, dtype=torch_mod.long)
    model = torch_mod.nn.Sequential(
        torch_mod.nn.Linear(32, 64),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(64, 3),
    ).to(device)
    opt = torch_mod.optim.SGD(model.parameters(), lr=0.08)
    loss_fn = torch_mod.nn.CrossEntropyLoss()
    rows: List[Dict[str, Any]] = []
    sync = _sync_device(torch_mod, device)

    if device == "cuda":
        try:
            torch_mod.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    for epoch in range(1, epochs + 1):
        perm = torch_mod.randperm(samples, device=device)
        if sync:
            sync()
        t0 = time.perf_counter()
        for start in range(0, samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = x[idx]
            yb = y[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        if sync:
            sync()
        epoch_ms = (time.perf_counter() - t0) * 1000

        with torch_mod.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
        max_memory_mb = ""
        if device == "cuda":
            try:
                max_memory_mb = round(torch_mod.cuda.max_memory_allocated() / (1024 * 1024), 2)
            except Exception:
                max_memory_mb = ""
        rows.append({
            "experiment": "mlp_train",
            "backend": backend,
            "device": _device_name(torch_mod, device),
            "backend_library": "pytorch",
            "available": True,
            "measurement_type": measurement_type,
            "dtype": "fp32",
            "epoch": epoch,
            "samples": samples,
            "batch_size": batch_size,
            "loss": round(float(loss.item()), 6),
            "accuracy": round(float(acc.item()), 6),
            "epoch_time_ms": round(epoch_ms, 4),
            "samples_per_second": round(samples / (epoch_ms / 1000.0), 2) if epoch_ms > 0 else "",
            "max_memory_allocated_mb": max_memory_mb,
            "torch_version": torch_mod.__version__,
            "torch_hip_version": getattr(torch_mod.version, "hip", ""),
            "note": f"Real {backend} MLP training via PyTorch",
        })
    return rows


def benchmark_mlp(epochs: int, samples: int, batch_size: int) -> List[Dict[str, Any]]:
    print("\n[3/3] MLP training benchmark ...")
    torch_mod = _import_torch()
    rocm_ok = _rocm_available(torch_mod)
    rows = _numpy_mlp_train(epochs=epochs, samples=samples, batch_size=batch_size)
    for row in rows:
        print(f"  [mlp] CPU epoch={row['epoch']:<2} loss={row['loss']:<8} acc={row['accuracy']:<6} time={row['epoch_time_ms']:>8.3f} ms")

    if torch_mod is not None:
        torch_cpu_rows = _torch_mlp_train(
            torch_mod,
            epochs=epochs,
            samples=samples,
            batch_size=batch_size,
            device="cpu",
            backend="Torch_CPU",
            measurement_type="real_torch_cpu",
        )
        for row in torch_cpu_rows:
            print(f"  [mlp] Torch_CPU epoch={row['epoch']:<2} loss={row['loss']:<8} acc={row['accuracy']:<6} time={row['epoch_time_ms']:>8.3f} ms")
        rows.extend(torch_cpu_rows)

    if not rocm_ok:
        rows.append({
            "experiment": "mlp_train",
            "backend": "ROCm_GPU",
            "device": "cuda",
            "backend_library": "pytorch",
            "available": False,
            "measurement_type": "unavailable",
            "dtype": "fp32",
            "epoch": "",
            "samples": samples,
            "batch_size": batch_size,
            "note": _rocm_note(torch_mod),
        })
        return rows

    assert torch_mod is not None
    gpu_rows = _torch_mlp_train(
        torch_mod,
        epochs=epochs,
        samples=samples,
        batch_size=batch_size,
        device="cuda",
        backend="ROCm_GPU",
        measurement_type="real_rocm_gpu",
    )
    for row in gpu_rows:
        print(f"  [mlp] ROCm_GPU epoch={row['epoch']:<2} loss={row['loss']:<8} acc={row['accuracy']:<6} time={row['epoch_time_ms']:>8.3f} ms")
    rows.extend(gpu_rows)
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic heterogeneous-computing course benchmarks")
    parser.add_argument("--matmul-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--precision-sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mlp-epochs", type=int, default=5)
    parser.add_argument("--mlp-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--results-dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    global RESULTS_DIR
    args = parse_args(argv)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    print("=" * 60)
    print("  Basic Heterogeneous Computing Benchmarks")
    print("=" * 60)
    print(f"  Matmul sizes: {args.matmul_sizes}")
    print(f"  Precision sizes: {args.precision_sizes}")
    print(f"  Repeats: {args.repeats}")
    print(f"  MLP epochs: {args.mlp_epochs}")

    matmul_rows = benchmark_matmul(args.matmul_sizes, args.repeats)
    precision_rows = benchmark_precision(args.precision_sizes, args.repeats)
    mlp_rows = benchmark_mlp(args.mlp_epochs, args.mlp_samples, args.batch_size)

    _write_csv(matmul_rows, "matmul_benchmark.csv")
    _write_csv(precision_rows, "precision_compare.csv")
    _write_csv(mlp_rows, "mlp_train_log.csv")

    print("\n" + "=" * 60)
    print("  Basic benchmarks complete")
    print("  ROCm_GPU rows are marked unavailable unless PyTorch HIP is detected.")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
