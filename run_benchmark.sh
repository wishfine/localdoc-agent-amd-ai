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

if [ ! -f "$SCRIPT_DIR/scripts/bootstrap_python_env.sh" ]; then
    error "未找到环境初始化脚本: scripts/bootstrap_python_env.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/bootstrap_python_env.sh"
bootstrap_python_env "$PYTHON" "$VENV_DIR"

# --- 安装依赖 ---
info "检查并安装依赖 ..."
pip install --quiet --upgrade pip
pip install --quiet numpy matplotlib psutil 2>/dev/null || warn "部分依赖安装失败"
info "依赖安装完成。"

# --- 创建目录 ---
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/figures"

# --- 参数处理 ---
RUN_BASIC=1
RUN_AGENT=1
RUN_VERTICAL=1
RUN_LLM=0
RUN_MONITOR=1
BASIC_ARGS=""
AGENT_ARGS=""
LLM_ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)
            BASIC_ARGS="--matmul-sizes 128 256 --precision-sizes 128 --repeats 2 --mlp-epochs 2 --mlp-samples 256 --batch-size 64"
            AGENT_ARGS="--doc-counts 1 --chunk-counts 10 --repeats 1"
            shift
            ;;
        --basic-only)
            RUN_AGENT=0
            RUN_VERTICAL=0
            RUN_LLM=0
            shift
            ;;
        --agent-only)
            RUN_BASIC=0
            shift
            ;;
        --with-llm)
            RUN_LLM=1
            shift
            ;;
        --allow-llm-hub)
            RUN_LLM=1
            LLM_ARGS="$LLM_ARGS --allow-hub"
            shift
            ;;
        --skip-vertical)
            RUN_VERTICAL=0
            shift
            ;;
        --no-monitor)
            RUN_MONITOR=0
            shift
            ;;
        *)
            AGENT_ARGS="$AGENT_ARGS $1"
            shift
            ;;
    esac
done

START_TIME=$(date +%s)

MONITOR_PID=""
MONITOR_STOP_FILE="$SCRIPT_DIR/results/.resource_monitor_stop"
cleanup_monitor() {
    if [ -n "${MONITOR_PID:-}" ]; then
        touch "$MONITOR_STOP_FILE"
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}
trap cleanup_monitor EXIT

if [ "$RUN_MONITOR" -eq 1 ]; then
    rm -f "$MONITOR_STOP_FILE"
    python "$SCRIPT_DIR/experiments/resource_monitor.py" \
        --results-dir "$SCRIPT_DIR/results" \
        --stop-file "$MONITOR_STOP_FILE" \
        --interval 1 \
        >/tmp/localdoc_resource_monitor.log 2>&1 &
    MONITOR_PID=$!
    info "资源/能效监控已启动 (pid=$MONITOR_PID)"
fi

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

# ====== 第 2 步: 基础实验 ======
echo ""
echo "============================================"
echo "  第 2 步: 基础异构实验"
echo "============================================"
echo ""

info "运行矩阵乘法、FP32/FP16 精度对比、MLP 训练实验 ..."
info "如检测到 ROCm PyTorch，将自动加入 ROCm_GPU 实测；否则仅记录 CPU 与 unavailable 状态"
echo ""

if [ "$RUN_BASIC" -eq 1 ]; then
    if python "$SCRIPT_DIR/experiments/basic_benchmarks.py" $BASIC_ARGS; then
        info "基础实验完成。"
    else
        error "基础实验执行失败！"
        exit 1
    fi
else
    warn "跳过基础实验 (--agent-only)"
fi

# ====== 第 3 步: 延迟基准测试 ======
echo ""
echo "============================================"
echo "  第 3 步: Agent 延迟基准测试 (自动检测硬件)"
echo "============================================"
echo ""

info "开始基准测试 ..."
info "如有真实 AMD GPU/NPU 硬件，将自动使用真实后端"
info "如无硬件，将使用 simulated 模式（所有数据标记为 simulated）"
echo ""

if [ "$RUN_AGENT" -eq 1 ]; then
    if python "$SCRIPT_DIR/experiments/benchmark_real.py" \
        $AGENT_ARGS; then
        info "基准测试完成。"
    else
        error "基准测试执行失败！"
        exit 1
    fi
else
    warn "跳过 Agent benchmark (--basic-only)"
fi

# ====== 第 4 步: 垂直行业 Demo ======
echo ""
echo "============================================"
echo "  第 4 步: 垂直行业应用流程"
echo "============================================"
echo ""

if [ "$RUN_VERTICAL" -eq 1 ]; then
    if python "$SCRIPT_DIR/experiments/demo_vertical_workflow.py" \
        --results-dir "$SCRIPT_DIR/results"; then
        info "垂直行业应用 transcript 已生成。"
    else
        warn "垂直行业应用流程失败，继续执行。"
    fi
else
    warn "跳过垂直行业应用流程"
fi

# ====== 第 5 步: 可选本地 LLM Benchmark ======
echo ""
echo "============================================"
echo "  第 5 步: 本地 LLM Benchmark (可选)"
echo "============================================"
echo ""

if [ "$RUN_LLM" -eq 1 ]; then
    if python "$SCRIPT_DIR/experiments/benchmark_llm_generation.py" \
        --results-dir "$SCRIPT_DIR/results" $LLM_ARGS; then
        info "本地 LLM benchmark 完成。"
    else
        warn "本地 LLM benchmark 失败或模型未准备好，继续执行。"
    fi

    if python "$SCRIPT_DIR/experiments/benchmark_rag_modes.py"; then
        info "RAG 模式对比 benchmark 完成。"
    else
        warn "RAG 模式对比失败或模型未准备好，继续执行。"
    fi
else
    warn "默认跳过 LLM benchmark；如需运行请传 --with-llm"
fi

# 停止能效监控，确保 power_trace.csv 可供绘图读取。
cleanup_monitor

# ====== 第 6 步: 生成图表 ======
echo ""
echo "============================================"
echo "  第 6 步: 生成结果图表"
echo "============================================"
echo ""

if python "$SCRIPT_DIR/experiments/plot_basic_results.py" \
    --results-dir "$SCRIPT_DIR/results" \
    --figures-dir "$SCRIPT_DIR/figures"; then
    info "基础实验图表生成完成。"
else
    warn "基础实验图表生成部分失败，但 CSV 结果已保存。"
fi

if python "$SCRIPT_DIR/experiments/plot_results.py" \
    --results-dir "$SCRIPT_DIR/results" \
    --figures-dir "$SCRIPT_DIR/figures"; then
    info "图表生成完成。"
else
    warn "图表生成部分失败，但 CSV 结果已保存。"
fi

if [ "$RUN_LLM" -eq 1 ]; then
    if python "$SCRIPT_DIR/experiments/plot_llm_results.py"; then
        info "LLM/RAG 图表生成完成。"
    else
        warn "LLM/RAG 图表生成部分失败，但 CSV 结果已保存。"
    fi
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

for csv in matmul_benchmark.csv precision_compare.csv mlp_train_log.csv latency_results.csv backend_results.csv resource_usage.csv power_trace.csv energy_summary.csv vertical_demo_transcript.csv llm_generation_benchmark.csv rag_mode_comparison.csv rag_stage_breakdown.csv; do
    if [ -f "$SCRIPT_DIR/results/$csv" ]; then
        LINES=$(wc -l < "$SCRIPT_DIR/results/$csv")
        echo "  📊 results/$csv  ($LINES 行)"
    fi
done

echo ""
for img in matmul_benchmark.png precision_compare.png mlp_training_curve.png energy_comparison.png latency_comparison.png backend_comparison.png resource_usage.png llm_generation_latency.png rag_mode_comparison.png rag_stage_breakdown.png; do
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
