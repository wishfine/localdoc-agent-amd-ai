#!/bin/bash
# 一键运行基准测试
# LocalDoc Agent - 异构资源调度基准测试
# 面向 AMD 锐龙 AI MAX+ 平台

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  运行异构资源调度基准测试"
echo "  LocalDoc Agent - AMD 锐龙 AI MAX+"
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

PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    error "未找到 Python！请安装 Python 3.9 或更高版本。"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
info "找到 Python: $PYTHON ($PY_VERSION)"

PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    error "Python 版本过低 ($PY_VERSION)，需要 3.9 或更高版本。"
    exit 1
fi
info "Python 版本检查通过: $PY_VERSION (>= 3.9)"

# --- 虚拟环境 ---
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    info "创建虚拟环境: $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    info "虚拟环境创建完成。"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "已激活虚拟环境: $(python --version)"

# --- 安装依赖 ---
info "检查并安装测试依赖 ..."

pip install --quiet --upgrade pip

# Install matplotlib for plotting
pip install --quiet matplotlib

# psutil is optional but nice to have
pip install --quiet psutil 2>/dev/null || warn "psutil 安装失败，资源监控功能将受限。"

info "依赖安装完成。"

# --- 创建目录 ---
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/figures"

# --- 参数处理 ---
BENCHMARK_ARGS=""
PLOT_ARGS=""

# Pass any extra CLI args to the benchmark script
if [ $# -gt 0 ]; then
    BENCHMARK_ARGS="$*"
fi

# --- 运行基准测试 ---
echo ""
echo "============================================"
echo "  第 1 步: 运行延迟基准测试"
echo "============================================"
echo ""

info "开始基准测试 (使用合成数据，完全离线) ..."
info "测试参数: $BENCHMARK_ARGS"
echo ""

START_TIME=$(date +%s)

python "$SCRIPT_DIR/experiments/benchmark_latency.py" \
    --output-dir "$SCRIPT_DIR/results" \
    $BENCHMARK_ARGS

BENCHMARK_EXIT=$?

if [ $BENCHMARK_EXIT -ne 0 ]; then
    error "基准测试执行失败 (退出码: $BENCHMARK_EXIT)"
    exit $BENCHMARK_EXIT
fi

BENCHMARK_END=$(date +%s)
BENCHMARK_DURATION=$((BENCHMARK_END - START_TIME))
info "基准测试完成，耗时 ${BENCHMARK_DURATION} 秒。"

# --- 生成图表 ---
echo ""
echo "============================================"
echo "  第 2 步: 生成结果图表"
echo "============================================"
echo ""

python "$SCRIPT_DIR/experiments/plot_results.py" \
    --results-dir "$SCRIPT_DIR/results" \
    --figures-dir "$SCRIPT_DIR/figures"

PLOT_EXIT=$?

if [ $PLOT_EXIT -ne 0 ]; then
    warn "图表生成部分失败 (退出码: $PLOT_EXIT)，但 CSV 结果已保存。"
fi

PLOT_END=$(date +%s)
TOTAL_DURATION=$((PLOT_END - START_TIME))

# --- 总结 ---
echo ""
echo "============================================"
echo "  基准测试完成！"
echo "============================================"
echo ""
info "总耗时: ${TOTAL_DURATION} 秒"
echo ""
info "生成的文件:"
echo "  CSV 结果:"

if [ -f "$SCRIPT_DIR/results/latency_results.csv" ]; then
    LINES=$(wc -l < "$SCRIPT_DIR/results/latency_results.csv")
    echo "    - results/latency_results.csv  ($LINES 行)"
fi
if [ -f "$SCRIPT_DIR/results/backend_results.csv" ]; then
    LINES=$(wc -l < "$SCRIPT_DIR/results/backend_results.csv")
    echo "    - results/backend_results.csv  ($LINES 行)"
fi
if [ -f "$SCRIPT_DIR/results/resource_usage.csv" ]; then
    LINES=$(wc -l < "$SCRIPT_DIR/results/resource_usage.csv")
    echo "    - results/resource_usage.csv   ($LINES 行)"
fi

echo ""
echo "  图表文件:"
for img in latency_comparison.png backend_comparison.png resource_usage.png; do
    if [ -f "$SCRIPT_DIR/figures/$img" ]; then
        SIZE=$(du -h "$SCRIPT_DIR/figures/$img" | cut -f1)
        echo "    - figures/$img  ($SIZE)"
    fi
done

echo ""
info "可使用以下命令查看图表:"
echo "  open $SCRIPT_DIR/figures/  # macOS"
echo "  xdg-open $SCRIPT_DIR/figures/  # Linux"
echo ""
