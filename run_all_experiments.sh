#!/bin/bash
# One-click full experiment runner for grading/demo submission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[信息]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*"; }

RUN_TESTS=1
RUN_LLM=1
ALLOW_LLM_HUB=0
BENCH_ARGS=()
ORIGINAL_ARGS="$*"

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)
            BENCH_ARGS+=("--quick")
            shift
            ;;
        --skip-tests)
            RUN_TESTS=0
            shift
            ;;
        --skip-llm)
            RUN_LLM=0
            shift
            ;;
        --allow-llm-hub)
            RUN_LLM=1
            ALLOW_LLM_HUB=1
            shift
            ;;
        --no-monitor|--basic-only|--agent-only|--skip-vertical)
            BENCH_ARGS+=("$1")
            shift
            ;;
        *)
            BENCH_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$SCRIPT_DIR/results" "$SCRIPT_DIR/figures"

LOG_FILE="$SCRIPT_DIR/results/full_experiment_run.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "============================================"
echo "  LocalDoc Agent - 一键全量实验"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    error "未找到 Python。"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    warn "检测到残缺虚拟环境: $VENV_DIR，删除后重新创建。"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    info "创建虚拟环境: $VENV_DIR"
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        error "创建虚拟环境失败。Ubuntu/Jupyter 环境通常需要先安装 python3-venv。"
        error "可尝试: sudo apt-get update && sudo apt-get install -y python3-venv"
        exit 1
    fi
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    error "虚拟环境仍不可用，缺少: $VENV_DIR/bin/activate"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Python: $(python --version)"

info "安装/检查基础依赖 ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy matplotlib psutil pytest

if [ "$RUN_TESTS" -eq 1 ]; then
    echo ""
    echo "============================================"
    echo "  Step 1/4: 单元测试"
    echo "============================================"
    python -m pytest -q
else
    warn "跳过单元测试 (--skip-tests)"
fi

echo ""
echo "============================================"
echo "  Step 2/4: 全量实验脚本"
echo "============================================"

if [ "$RUN_LLM" -eq 1 ]; then
    BENCH_ARGS+=("--with-llm")
    if [ "$ALLOW_LLM_HUB" -eq 1 ]; then
        BENCH_ARGS+=("--allow-llm-hub")
    fi
else
    warn "跳过本地 LLM benchmark (--skip-llm)"
fi

bash "$SCRIPT_DIR/run_benchmark.sh" "${BENCH_ARGS[@]}"

echo ""
echo "============================================"
echo "  Step 3/4: 生成实验结果清单"
echo "============================================"

MANIFEST="$SCRIPT_DIR/results/experiment_manifest.txt"
{
    echo "LocalDoc Agent full experiment manifest"
    echo "Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Command: bash run_all_experiments.sh $ORIGINAL_ARGS"
    echo ""
    echo "[Environment]"
    echo "Python: $(python --version)"
    echo "CWD: $SCRIPT_DIR"
    echo ""
    echo "[Key result files]"
    for path in \
        results/environment_report.txt \
        results/rocminfo.txt \
        results/rocm_smi.txt \
        results/hipcc_version.txt \
        results/hipconfig_full.txt \
        results/matmul_benchmark.csv \
        results/precision_compare.csv \
        results/mlp_train_log.csv \
        results/latency_results.csv \
        results/backend_results.csv \
        results/resource_usage.csv \
        results/power_trace.csv \
        results/energy_summary.csv \
        results/vertical_demo_transcript.csv \
        results/llm_generation_benchmark.csv \
        figures/matmul_benchmark.png \
        figures/precision_compare.png \
        figures/mlp_training_curve.png \
        figures/energy_comparison.png \
        figures/latency_comparison.png \
        figures/backend_comparison.png \
        figures/resource_usage.png
    do
        if [ -f "$SCRIPT_DIR/$path" ]; then
            printf "OK  %s\n" "$path"
        else
            printf "MISS %s\n" "$path"
        fi
    done
    echo ""
    echo "[Git status]"
    git status --short || true
} > "$MANIFEST"

info "结果清单: $MANIFEST"
info "完整运行日志: $LOG_FILE"

echo ""
echo "============================================"
echo "  Step 4/4: 截图清单"
echo "============================================"
echo "截图建议见: $SCRIPT_DIR/docs/screenshot_checklist.md"
echo ""
sed -n '1,220p' "$SCRIPT_DIR/docs/screenshot_checklist.md"

echo ""
echo "============================================"
echo "  全量实验完成"
echo "============================================"
info "建议先截图 docs/screenshot_checklist.md 中标为 [必截] 的项目。"
