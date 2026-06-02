"""
Local LLM Backend for LocalDoc Agent.

Uses Qwen3-1.7B via Hugging Face Transformers for local answer generation.

Important:
- This is a local inference backend. It does NOT call any cloud API.
- It is optional and only activated when LOCALDOC_USE_LLM=1.
- Embedding still falls back to CPUBackend TF-IDF to keep the system simple.
- Without real AMD GPU/NPU, inference runs on CPU only.
- This is NOT an AMD GPU/NPU hardware benchmark.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from localdoc.backends.cpu_backend import CPUBackend
from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class LocalLLMBackend:
    """
    Local LLM generation backend using Qwen3-1.7B.

    Implements:
    - embed_texts(): delegates to CPUBackend TF-IDF (no separate embedding model)
    - generate_answer(): uses local Qwen3-1.7B for generation

    Qwen3 uses thinking mode by default. This backend disables it with
    enable_thinking=False for stable, fast demo inference.

    Environment variables:
    - LOCALDOC_USE_LLM=1: enable this backend
    - LOCALDOC_LLM_MODEL_PATH: path to local model directory
    - LOCALDOC_LLM_MODEL_ID: Hugging Face model ID (fallback if local path not found)
    - LOCALDOC_LLM_MAX_NEW_TOKENS: max generation tokens (default 128)
    - LOCALDOC_LLM_CONTEXT_CHARS: max context characters (default 1600)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_id: str = "Qwen/Qwen3-1.7B",
        max_new_tokens: int = 128,
        context_chars: int = 1600,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_model_path = project_root / "models" / "qwen3-1.7b"

        self.model_path = str(
            model_path or os.getenv("LOCALDOC_LLM_MODEL_PATH", default_model_path)
        )
        self.model_id = os.getenv("LOCALDOC_LLM_MODEL_ID", model_id)
        self.max_new_tokens = int(
            os.getenv("LOCALDOC_LLM_MAX_NEW_TOKENS", str(max_new_tokens))
        )
        self.context_chars = int(
            os.getenv("LOCALDOC_LLM_CONTEXT_CHARS", str(context_chars))
        )

        self._cpu_backend = CPUBackend()
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_time_s: float = 0.0

    @property
    def name(self) -> str:
        return "LocalLLM(Qwen3-1.7B)"

    def is_available(self) -> bool:
        """Check if the model directory exists locally."""
        model_dir = Path(self.model_path)
        return model_dir.exists() and (model_dir / "config.json").exists()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding: delegate to CPUBackend TF-IDF.
        Keeps embedding lightweight; no separate embedding model needed.
        """
        return self._cpu_backend.embed_texts(texts)

    def _lazy_load(self) -> None:
        """Load the model on first use (lazy loading)."""
        if self._loaded:
            return

        t0 = time.perf_counter()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch

            # Determine model source: local path or Hugging Face Hub
            if self.is_available():
                model_source = self.model_path
                logger.info("Loading local LLM from: %s", model_source)
            else:
                model_source = self.model_id
                logger.info(
                    "Local model not found at %s, loading from HF Hub: %s",
                    self.model_path,
                    model_source,
                )

            # Device and dtype selection
            hip_version = getattr(torch.version, "hip", None)
            cuda_version = getattr(torch.version, "cuda", None)

            if hip_version and torch.cuda.is_available():
                self._device = "cuda"
                torch_dtype = torch.float16
                logger.info("Using AMD ROCm GPU (HIP %s) for LLM inference", hip_version)
            elif cuda_version and torch.cuda.is_available():
                self._device = "cuda"
                torch_dtype = torch.float16
                logger.info("Using CUDA GPU (CUDA %s) for LLM inference", cuda_version)
            else:
                self._device = "cpu"
                torch_dtype = torch.float32
                logger.info("Using CPU for LLM inference (no GPU detected)")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True
            self._load_time_s = round(time.perf_counter() - t0, 2)
            logger.info(
                "Local LLM loaded in %.2fs (device=%s, dtype=%s)",
                self._load_time_s,
                self._device,
                torch_dtype,
            )

        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            logger.error("Failed to load local LLM: %s", self._load_error)
            raise

    def generate_answer(
        self,
        query: str,
        context: str,
        max_length: int = 512,
    ) -> str:
        """
        Generate an answer using the local LLM.

        Disables Qwen3 thinking mode (enable_thinking=False) for stable demo.
        Uses do_sample=True with temperature=0.7, top_p=0.8, top_k=20
        as recommended by Qwen3 for non-thinking mode.

        Args:
            query: user question
            context: formatted context string
            max_length: kept for API compatibility

        Returns:
            Generated answer text
        """
        self._lazy_load()

        if not self._loaded:
            return "本地 LLM 加载失败，无法生成回答。请检查模型文件和依赖。"

        context = context or ""
        context = context[: self.context_chars]

        system_prompt = (
            "你是一个本地知识库问答助手。"
            "请只根据给定的文档内容回答问题。"
            "如果文档中没有相关信息，请明确说“文档中没有找到相关信息”。"
            "回答要简洁，优先使用中文。"
        )

        user_prompt = (
            "下面是从本地知识库检索到的文档片段：\n\n"
            f"{context}\n\n"
            "用户问题：\n"
            f"{query}\n\n"
            "请根据上述文档片段回答用户问题。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        # Apply chat template with thinking disabled for Qwen3
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer([text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated tokens (not the input)
        output_ids = generated_ids[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        if not response:
            return "文档中没有找到相关信息。"

        return response

    def get_device_info(self) -> Dict[str, Any]:
        """Get backend device and configuration info."""
        hip_version = None
        cuda_version = None
        cuda_available = False
        try:
            import torch
            hip_version = getattr(torch.version, "hip", None)
            cuda_version = getattr(torch.version, "cuda", None)
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        is_rocm = bool(hip_version and cuda_available)
        is_cuda = bool(cuda_version and cuda_available)

        # Determine hardware note
        if is_rocm:
            hardware_note = (
                "AMD ROCm runtime detected (HIP %s); "
                "LLM benchmark is local inference only, not strict AMD hardware benchmark."
                % hip_version
            )
        elif is_cuda:
            hardware_note = "CUDA GPU detected (not AMD ROCm); not AMD hardware benchmark"
        else:
            hardware_note = "CPU inference only; not AMD GPU/NPU hardware benchmark."

        return {
            "backend": self.name,
            "model_path": self.model_path,
            "model_id": self.model_id,
            "model_available_local": self.is_available(),
            "device": self._device,
            "loaded": self._loaded,
            "load_error": self._load_error,
            "load_time_s": self._load_time_s,
            "max_new_tokens": self.max_new_tokens,
            "context_chars": self.context_chars,
            "is_local_llm": True,
            "is_amd_hardware_benchmark": False,
            "is_rocm_runtime_detected": is_rocm,
            "is_cuda_runtime_detected": is_cuda,
            "torch_cuda_available": cuda_available,
            "torch_hip_version": hip_version,
            "torch_cuda_version": cuda_version,
            "hardware_note": hardware_note,
            "note": "Local LLM inference only; not AMD GPU/NPU hardware benchmark.",
        }
