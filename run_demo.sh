#!/bin/bash
# 一键运行 Gradio Web Demo
# LocalDoc Agent - 本地知识库智能体
# 面向 AMD 锐龙 AI MAX+ 平台

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  LocalDoc Agent - 本地知识库智能体"
echo "  面向 AMD 锐龙 AI MAX+ 平台"
echo "============================================"
echo ""

# --- 颜色定义 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[信息]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*"; }

# --- 检查 Python 版本 ---
info "检查 Python 环境 ..."

# Prefer python3, fallback to python
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    error "未找到 Python！请安装 Python 3.9 或更高版本。"
    error "  macOS: brew install python@3.11"
    error "  Ubuntu: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
info "找到 Python: $PYTHON ($PY_VERSION)"

# Check version >= 3.9
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    error "Python 版本过低 ($PY_VERSION)，需要 3.9 或更高版本。"
    exit 1
fi
info "Python 版本检查通过: $PY_VERSION (>= 3.9)"

# --- 虚拟环境 ---
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    warn "检测到残缺虚拟环境: $VENV_DIR，删除后重新创建。"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    info "创建虚拟环境: $VENV_DIR ..."
    if ! $PYTHON -m venv "$VENV_DIR"; then
        error "创建虚拟环境失败。Ubuntu/Jupyter 环境通常需要先安装 python3-venv。"
        error "可尝试: sudo apt-get update && sudo apt-get install -y python3-venv"
        exit 1
    fi
    info "虚拟环境创建完成。"
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    error "虚拟环境仍不可用，缺少: $VENV_DIR/bin/activate"
    exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "已激活虚拟环境: $(python --version)"

# --- 安装依赖 ---
info "检查并安装依赖 ..."

pip install --quiet --upgrade pip

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    info "从 requirements.txt 安装依赖 ..."
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
    info "依赖安装完成。"
else
    warn "未找到 requirements.txt，尝试安装基础依赖 ..."
    pip install --quiet gradio numpy
fi

# --- 检查 Gradio 是否可用 ---
if ! python -c "import gradio" 2>/dev/null; then
    warn "Gradio 未安装，正在安装 ..."
    pip install --quiet gradio
fi

# --- 启动 Gradio 应用 ---
APP_FILE="localdoc/app.py"

if [ ! -f "$SCRIPT_DIR/$APP_FILE" ]; then
    error "未找到应用入口文件: $APP_FILE"
    exit 1
fi

info "启动 Gradio Web Demo ..."
info "访问地址: http://localhost:7860"
echo ""

python "$APP_FILE"
