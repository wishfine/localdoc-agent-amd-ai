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
  --rocm       安装 ROCm 版 PyTorch。必须显式设置 LOCALDOC_TORCH_ROCM_INDEX_URL，
               避免在 AMD/Jupyter 平台误装与容器不匹配的 ROCm wheel。
  --cpu        安装 CPU 版 PyTorch，用于无 AMD ROCm 的普通环境。
  --skip-torch 只安装不会触发 torch 解析的通用 LLM 依赖，不安装 PyTorch。

默认行为:
  为避免在 AMD 平台误装 CUDA 版 PyTorch，未传参数时等同于 --skip-torch。

常用:
  # AMD/Jupyter 平台已有可用 ROCm PyTorch 时
  LOCALDOC_USE_CURRENT_PYTHON=1 bash scripts/setup_llm.sh --skip-torch

  # 只有明确知道匹配的 ROCm wheel index 时才安装 torch
  LOCALDOC_TORCH_ROCM_INDEX_URL=<matching-rocm-wheel-index> bash scripts/setup_llm.sh --rocm

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

install_common_deps_without_torch() {
    info "安装基础依赖 ..."
    pip install -r requirements.txt

    info "安装 LLM 通用依赖（先不安装 accelerate，避免自动拉取默认 torch）..."
    pip install \
        "transformers>=4.51.0" \
        "safetensors>=0.4.2" \
        "huggingface_hub>=0.23.0"
}

install_accelerate_after_torch() {
    if [ "$TORCH_MODE" = "skip" ]; then
        warn "以 --no-deps 安装 accelerate，避免它自动安装 PyPI 默认 torch。"
        pip install --no-deps "accelerate>=0.26.0"
    else
        info "安装 accelerate；此时 PyTorch 已由 $TORCH_MODE 模式显式安装。"
        pip install "accelerate>=0.26.0"
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

install_common_deps_without_torch

ROCM_INDEX_URL="${LOCALDOC_TORCH_ROCM_INDEX_URL:-}"
CPU_INDEX_URL="${LOCALDOC_TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

case "$TORCH_MODE" in
    rocm)
        if [ -z "$ROCM_INDEX_URL" ]; then
            error "--rocm 需要显式设置 LOCALDOC_TORCH_ROCM_INDEX_URL。"
            error "当前 AMD/Jupyter 平台如已有可用 ROCm PyTorch，推荐改用:"
            error "  LOCALDOC_USE_CURRENT_PYTHON=1 bash scripts/setup_llm.sh --skip-torch"
            exit 2
        fi
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
        warn "未安装 PyTorch。AMD/Jupyter 平台如已有可用 ROCm PyTorch，这是推荐模式。"
        warn "如确需安装 torch，请显式选择:"
        warn "  AMD ROCm: LOCALDOC_TORCH_ROCM_INDEX_URL=<matching-index> bash scripts/setup_llm.sh --rocm"
        warn "  CPU only: bash scripts/setup_llm.sh --cpu"
        ;;
esac

install_accelerate_after_torch

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
    print("[错误] 当前 torch.version.hip 为空，不是 ROCm 版 PyTorch。")
    print("[错误] 安装已停止。请检查 LOCALDOC_TORCH_ROCM_INDEX_URL 或 PyTorch/ROCm/Python 版本是否匹配。")
    raise SystemExit(2)
if mode == "rocm" and not available:
    print("[警告] 当前 PyTorch 没有检测到可用 GPU；请检查 ROCm 驱动、权限和 wheel 版本。")
if mode == "cpu" and cuda:
    print("[错误] --cpu 模式下检测到 CUDA 版 PyTorch，安装不符合预期。")
    raise SystemExit(3)
if cuda and not hip:
    print("[错误] 检测到 CUDA 版 PyTorch。这不能作为 AMD ROCm GPU 实测。")
    raise SystemExit(4)
PY

echo ""
echo "[完成] LLM 依赖处理完成"
echo ""
echo "下一步：下载模型"
echo "  bash scripts/download_llm.sh"
