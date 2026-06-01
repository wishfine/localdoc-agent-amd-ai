#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  LocalDoc Agent - Setup Local LLM"
echo "  Model: Qwen2.5-0.5B-Instruct"
echo "============================================"

PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "[错误] 未找到 Python"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[信息] 创建虚拟环境: $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[信息] Python: $(python --version)"
echo "[信息] 升级 pip"
pip install --upgrade pip

echo "[信息] 安装 LLM 可选依赖 (torch, transformers, accelerate, ...)"
pip install -r requirements-llm.txt

echo ""
echo "[完成] LLM 依赖安装完成"
echo ""
echo "下一步：下载模型"
echo "  bash scripts/download_llm.sh"
