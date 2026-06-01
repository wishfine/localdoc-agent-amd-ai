#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "============================================"
echo "  LocalDoc Agent - LLM Benchmark"
echo "  Model: Qwen3-1.7B"
echo "  ⚠️ Local LLM inference, NOT AMD hardware benchmark"
echo "============================================"

mkdir -p results figures

# Step 1: LLM generation latency benchmark
echo ""
echo "[1/3] LLM 生成延迟基准测试 ..."
python experiments/benchmark_llm_generation.py

# Step 2: RAG mode comparison
echo ""
echo "[2/3] RAG 模式对比 (extractive vs local LLM) ..."
python experiments/benchmark_rag_modes.py

# Step 3: Plot results
echo ""
echo "[3/3] 生成图表 ..."
python experiments/plot_llm_results.py

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
