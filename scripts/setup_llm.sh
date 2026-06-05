#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

info() {
    echo "[信息] $1"
}

warn() {
    echo "[警告] $1"
}

error() {
    echo "[错误] $1"
}

usage() {
    cat <<'EOF'
用法:
  bash scripts/setup_llm.sh [--rocm | --cpu | --skip-torch]

说明:
  --rocm       安装 ROCm 版 PyTorch。默认索引为 https://download.pytorch.org/whl/rocm6.4
               可用 LOCALDOC_TORCH_ROCM_INDEX_URL 覆盖。
  --cpu        安装 CPU 版 PyTorch，用于无 AMD ROCm 的普通环境。
  --skip-torch 只安装 transformers/accelerate 等通用 LLM 依赖，不安装 PyTorch。

默认行为:
  为避免在 AMD 平台误装 CUDA 版 PyTorch，未传参数时等同于 --skip-torch。

常用:
  # AMD ROCm 平台
  bash scripts/setup_llm.sh --rocm

  # 普通 CPU 环境
  bash scripts/setup_llm.sh --cpu
EOF
}

remove_cuda_wheel_leftovers() {
    local cuda_pkgs
    cuda_pkgs="$(python - <<'PY'
import subprocess
import sys

try:
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    raise SystemExit(0)

names = []
for line in output.splitlines():
    if "==" not in line:
        continue
    name = line.split("==", 1)[0]
    lower = name.lower()
    if lower.startswith("nvidia-"):
        names.append(name)

print(" ".join(names))
PY
)"

    if [ -n "$cuda_pkgs" ]; then
        warn "清理 CUDA wheel 残留依赖: $cuda_pkgs"
        # shellcheck disable=SC2086
        pip uninstall -y $cuda_pkgs || true
    fi
}

TORCH_MODE="skip"
while [ $# -gt 0 ]; do
    case "$1" in
        --rocm)
            TORCH_MODE="rocm"
            ;;
        --cpu)
            TORCH_MODE="cpu"
            ;;
        --skip-torch)
            TORCH_MODE="skip"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

echo "============================================"
echo "  LocalDoc Agent - Setup Local LLM"
echo "  Model: Qwen3-1.7B"
echo "============================================"

PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    error "未找到 Python"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

# shellcheck source=scripts/bootstrap_python_env.sh
source "$SCRIPT_DIR/scripts/bootstrap_python_env.sh"
bootstrap_python_env "$PYTHON" "$VENV_DIR"

info "Python: $(python --version)"
info "升级 pip"
pip install --upgrade pip

info "安装 LLM 通用依赖 (不包含 torch) ..."
pip install -r requirements-llm.txt

ROCM_INDEX_URL="${LOCALDOC_TORCH_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm6.4}"
CPU_INDEX_URL="${LOCALDOC_TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

case "$TORCH_MODE" in
    rocm)
        info "安装 ROCm 版 PyTorch: $ROCM_INDEX_URL"
        warn "将先卸载现有 torch/torchvision/torchaudio，避免 CUDA 版残留。"
        pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm triton || true
        remove_cuda_wheel_leftovers
        pip install --index-url "$ROCM_INDEX_URL" torch torchvision torchaudio
        ;;
    cpu)
        info "安装 CPU 版 PyTorch: $CPU_INDEX_URL"
        pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm triton || true
        remove_cuda_wheel_leftovers
        pip install --index-url "$CPU_INDEX_URL" torch torchvision torchaudio
        ;;
    skip)
        warn "未安装 PyTorch。为避免误装 CUDA 版，请显式选择:"
        warn "  AMD ROCm: bash scripts/setup_llm.sh --rocm"
        warn "  CPU only: bash scripts/setup_llm.sh --cpu"
        ;;
esac

LOCALDOC_TORCH_MODE="$TORCH_MODE" python - <<'PY'
import os

mode = os.getenv("LOCALDOC_TORCH_MODE", "skip")
try:
    import torch
except Exception as exc:
    print("[检查] torch: 未安装或不可导入")
    print(f"[检查] 原因: {exc}")
    raise SystemExit(0)

hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
available = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if available else None

print(f"[检查] torch: {torch.__version__}")
print(f"[检查] torch.version.hip: {hip}")
print(f"[检查] torch.version.cuda: {cuda}")
print(f"[检查] torch.cuda.is_available(): {available}")
print(f"[检查] device: {device}")

if mode == "rocm" and not hip:
    print("[警告] 当前 torch.version.hip 为空，不是 ROCm 版 PyTorch。")
if mode == "rocm" and not available:
    print("[警告] 当前 PyTorch 没有检测到可用 GPU；请检查 ROCm 驱动、权限和 wheel 版本。")
if cuda and not hip:
    print("[警告] 检测到 CUDA 版 PyTorch。这不能作为 AMD ROCm GPU 实测。")
PY

echo ""
echo "[完成] LLM 依赖处理完成"
echo ""
echo "下一步：下载模型"
echo "  bash scripts/download_llm.sh"
