"""
Download Qwen3-1.7B model from Hugging Face Hub.

Run: python scripts/download_llm.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-1.7B"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = PROJECT_ROOT / "models" / "qwen3-1.7b"


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LocalDoc Agent - Download Local LLM")
    print(f"Model: {MODEL_ID}")
    print(f"Target: {LOCAL_DIR}")
    print("=" * 60)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("\n下载完成。")
    print(f"模型目录: {LOCAL_DIR}")
    print("\n测试命令:")
    print("  python scripts/test_llm.py")


if __name__ == "__main__":
    main()
