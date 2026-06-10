#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

LLM_ARGS=()
RAG_ARGS=()
REQUIRE_GPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --require-gpu|--require-llm-gpu)
            REQUIRE_GPU=1
            export LOCALDOC_REQUIRE_LLM_GPU=1
            LLM_ARGS+=("--require-gpu")
            RAG_ARGS+=("--require-gpu")
            shift
            ;;
        --allow-hub|--allow-llm-hub)
            LLM_ARGS+=("--allow-hub")
            shift
            ;;
        *)
            LLM_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -f "$SCRIPT_DIR/scripts/bootstrap_python_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/scripts/bootstrap_python_env.sh"
fi

if [ "$REQUIRE_GPU" -eq 1 ]; then
    if command -v python3 >/dev/null 2>&1; then
        localdoc_prefer_current_python_for_rocm "python3" "$SCRIPT_DIR" || true
    fi
fi

if [ -n "${LOCALDOC_USE_CURRENT_PYTHON:-}" ] && declare -F bootstrap_python_env >/dev/null 2>&1; then
    bootstrap_python_env "python3" "$SCRIPT_DIR/.venv"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "[错误] 未找到 Python。请确认 python3 可用。"
    exit 1
fi

echo "============================================"
echo "  LocalDoc Agent - LLM Benchmark"
echo "  Model: Qwen3-1.7B"
echo "  ⚠️ Local LLM inference, NOT AMD hardware benchmark"
echo "  Require GPU: ${LOCALDOC_REQUIRE_LLM_GPU:-0}"
echo "============================================"

mkdir -p results figures

# Step 1: LLM generation latency benchmark
echo ""
echo "[1/3] LLM 生成延迟基准测试 ..."
"$PYTHON" experiments/benchmark_llm_generation.py "${LLM_ARGS[@]}"

# Step 2: RAG mode comparison
echo ""
echo "[2/3] RAG 模式对比 (extractive vs local LLM) ..."
"$PYTHON" experiments/benchmark_rag_modes.py "${RAG_ARGS[@]}"

# Step 3: Plot results
echo ""
echo "[3/3] 生成图表 ..."
"$PYTHON" experiments/plot_llm_results.py

echo ""
echo "============================================"
echo "  LLM Benchmark 完成！"
echo "============================================"
echo ""
echo "  CSV 结果:"
for f in llm_generation_benchmark.csv rag_mode_comparison.csv rag_stage_breakdown.csv; do
    [ -f "results/$f" ] && echo "    results/$f"
done
echo ""
echo "  图表:"
for f in llm_generation_latency.png rag_mode_comparison.png rag_stage_breakdown.png; do
    [ -f "figures/$f" ] && echo "    figures/$f"
done
