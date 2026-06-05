"""
Download a local LLM model from Hugging Face Hub.

Run: python scripts/download_llm.py
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PRESETS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", PROJECT_ROOT / "models" / "qwen3-1.7b"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", PROJECT_ROOT / "models" / "qwen2.5-0.5b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local LLM for LocalDoc Agent")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="qwen3-1.7b")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--local-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    preset_model_id, preset_dir = MODEL_PRESETS[args.preset]
    model_id = args.model_id or preset_model_id
    local_dir = Path(args.local_dir) if args.local_dir else preset_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LocalDoc Agent - Download Local LLM")
    print(f"Preset: {args.preset}")
    print(f"Model: {model_id}")
    print(f"Target: {local_dir}")
    print("=" * 60)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("\n下载完成。")
    print(f"模型目录: {local_dir}")
    print("\n测试命令:")
    print(f"  LOCALDOC_LLM_MODEL_PATH={local_dir} python scripts/test_llm.py")


if __name__ == "__main__":
    main()
