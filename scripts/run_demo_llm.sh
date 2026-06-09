#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[信息]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*"; }

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    error "未找到 Python。"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/scripts/bootstrap_python_env.sh" ]; then
    error "未找到环境初始化脚本: scripts/bootstrap_python_env.sh"
    exit 1
fi

REQUIRE_GPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --require-gpu|--require-llm-gpu)
            REQUIRE_GPU=1
            export LOCALDOC_REQUIRE_LLM_GPU=1
            shift
            ;;
        *)
            warn "忽略未知参数: $1"
            shift
            ;;
    esac
done

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/bootstrap_python_env.sh"
bootstrap_python_env "$PYTHON" "$SCRIPT_DIR/.venv"

info "检查并安装依赖 ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

export LOCALDOC_USE_LLM=1
export LOCALDOC_LLM_MODEL_PATH="$SCRIPT_DIR/models/qwen3-1.7b"
export LOCALDOC_LLM_MAX_NEW_TOKENS=128
export LOCALDOC_LLM_CONTEXT_CHARS=1600

echo "============================================"
echo "  LocalDoc Agent - Demo with Local LLM"
echo "  Model: Qwen3-1.7B (local, no cloud API)"
echo "  Thinking mode: disabled (enable_thinking=False)"
echo "============================================"
echo ""
echo "  LOCALDOC_USE_LLM=$LOCALDOC_USE_LLM"
echo "  LOCALDOC_LLM_MODEL_PATH=$LOCALDOC_LLM_MODEL_PATH"
echo "  LOCALDOC_LLM_MAX_NEW_TOKENS=$LOCALDOC_LLM_MAX_NEW_TOKENS"
echo "  LOCALDOC_REQUIRE_LLM_GPU=${LOCALDOC_REQUIRE_LLM_GPU:-0}"
echo "  Require GPU: $REQUIRE_GPU"
echo ""

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -m localdoc.app
