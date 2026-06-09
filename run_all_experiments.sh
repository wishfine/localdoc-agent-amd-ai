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
        --require-llm-gpu)
            RUN_LLM=1
            BENCH_ARGS+=("--require-llm-gpu")
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
if [ ! -f "$SCRIPT_DIR/scripts/bootstrap_python_env.sh" ]; then
    error "未找到环境初始化脚本: scripts/bootstrap_python_env.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/bootstrap_python_env.sh"
bootstrap_python_env "$PYTHON" "$VENV_DIR"
info "Python: $(python --version)"

info "安装/检查基础依赖 ..."
pip install --quiet --upgrade pip
pip install --quiet numpy matplotlib psutil pytest

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
        results/rocm_tools_summary.csv \
        results/rocm_tuning_recommendations.md \
        results/amd_smi_list.txt \
        results/amd_smi_static.txt \
        results/amd_smi_metric.txt \
        results/rocm_smi_performance.txt \
        results/rocm_bandwidth_test.txt \
        results/rocprofiler_tools.txt \
        results/rocprofiler_run.txt \
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
        results/rag_mode_comparison.csv \
        results/rag_stage_breakdown.csv \
        results/full_experiment_run.log \
        figures/matmul_benchmark.png \
        figures/precision_compare.png \
        figures/mlp_training_curve.png \
        figures/energy_comparison.png \
        figures/latency_comparison.png \
        figures/backend_comparison.png \
        figures/resource_usage.png \
        figures/llm_generation_latency.png \
        figures/rag_mode_comparison.png \
        figures/rag_stage_breakdown.png
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
