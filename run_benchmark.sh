#!/bin/bash
# 一键运行基准测试
# LocalDoc Agent - 异构资源调度仿真实验
# 面向 AMD 锐龙 AI MAX+ 平台

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  运行异构资源调度基准测试"
echo "  LocalDoc Agent - AMD 锐龙 AI MAX+"
echo "  自动检测硬件：有真实 GPU/NPU 则实测，否则 simulated"
echo "============================================"
echo ""

# --- 颜色定义 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[信息]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*"; }

# --- 检查 Python ---
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
info "检查并安装依赖 ..."
pip install --quiet --upgrade pip
pip install --quiet matplotlib psutil 2>/dev/null || warn "部分依赖安装失败"
info "依赖安装完成。"

# --- 创建目录 ---
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/figures"

# --- 参数处理 ---
BENCHMARK_ARGS=""
if [ $# -gt 0 ]; then
    BENCHMARK_ARGS="$*"
fi

START_TIME=$(date +%s)

# ====== 第 1 步: 环境检查 ======
echo ""
echo "============================================"
echo "  第 1 步: 环境检查"
echo "============================================"
echo ""

if python "$SCRIPT_DIR/experiments/check_environment.py"; then
    info "环境检查完成。"
else
    warn "环境检查脚本出错，继续执行后续步骤。"
fi

# ====== 第 2 步: 延迟基准测试 ======
echo ""
echo "============================================"
echo "  第 2 步: 延迟基准测试 (自动检测硬件)"
echo "============================================"
echo ""

info "开始基准测试 ..."
info "如有真实 AMD GPU/NPU 硬件，将自动使用真实后端"
info "如无硬件，将使用 simulated 模式（所有数据标记为 simulated）"
echo ""

if python "$SCRIPT_DIR/experiments/benchmark_real.py" \
    $BENCHMARK_ARGS; then
    info "基准测试完成。"
else
    error "基准测试执行失败！"
    exit 1
fi

# ====== 第 3 步: 生成图表 ======
echo ""
echo "============================================"
echo "  第 3 步: 生成结果图表"
echo "============================================"
echo ""

if python "$SCRIPT_DIR/experiments/plot_results.py" \
    --results-dir "$SCRIPT_DIR/results" \
    --figures-dir "$SCRIPT_DIR/figures"; then
    info "图表生成完成。"
else
    warn "图表生成部分失败，但 CSV 结果已保存。"
fi

# ====== 总结 ======
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "  实验完成！"
echo "============================================"
echo ""
info "总耗时: ${TOTAL_DURATION} 秒"
echo ""
info "生成的文件:"
echo ""

if [ -f "$SCRIPT_DIR/results/environment_report.txt" ]; then
    echo "  📋 results/environment_report.txt"
fi

for csv in latency_results.csv backend_results.csv resource_usage.csv; do
    if [ -f "$SCRIPT_DIR/results/$csv" ]; then
        LINES=$(wc -l < "$SCRIPT_DIR/results/$csv")
        echo "  📊 results/$csv  ($LINES 行)"
    fi
done

echo ""
for img in latency_comparison.png backend_comparison.png resource_usage.png; do
    if [ -f "$SCRIPT_DIR/figures/$img" ]; then
        SIZE=$(du -h "$SCRIPT_DIR/figures/$img" | cut -f1)
        echo "  📈 figures/$img  ($SIZE)"
    fi
done

echo ""
info "可使用以下命令查看图表:"
echo "  open $SCRIPT_DIR/figures/  # macOS"
echo "  xdg-open $SCRIPT_DIR/figures/  # Linux"
echo ""
