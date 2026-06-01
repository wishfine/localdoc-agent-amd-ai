#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

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
echo ""

python localdoc/app.py
