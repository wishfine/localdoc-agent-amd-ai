# 本地 LLM 接入指南

## 为什么选择 Qwen3-1.7B

| 特征 | 说明 |
|------|------|
| 模型 | Qwen3-1.7B（1.7B 参数） |
| 语言支持 | 中文、英文 |
| 许可证 | Apache-2.0（Hugging Face） |
| 依赖 | transformers>=4.51.0 + 显式选择的 PyTorch（ROCm 或 CPU），不需要 vLLM/Ollama |
| 用途 | 课程展示本地 LLM 推理能力 |
| 特殊设置 | 关闭 thinking mode（enable_thinking=False） |

## 重要声明

- 该 LLM 是**可选后端**，不影响默认的 CPU fallback 抽取式回答。
- 该 LLM **完全本地运行**，不调用任何云端 API。
- 如果没有真实 AMD GPU/NPU 硬件，模型在 **CPU 上推理**，不代表 GPU/NPU 实测。
- 该实验只证明"本地 LLM 生成链路可运行"，不声称 AMD GPU/NPU 加速。
- Qwen3 默认开启 thinking mode，本项目关闭它（enable_thinking=False）以保证演示稳定。
- 不要在 AMD 平台直接运行 `pip install torch` 或 `pip install -r requirements-llm.txt`。后者虽然没有直接写 `torch`，但 `accelerate` 会通过传递依赖拉取默认 PyPI torch，仍可能装成 CUDA 版。
- 必须通过 `scripts/setup_llm.sh --rocm` 安装；脚本会先从 ROCm wheel 源安装 PyTorch，再安装 `accelerate`。

## 安装步骤

### 1. 安装依赖

AMD ROCm 平台：

```bash
bash scripts/setup_llm.sh --rocm
```

普通 CPU 环境：

```bash
bash scripts/setup_llm.sh --cpu
```

默认不传参数时，`setup_llm.sh` 只安装不会触发 torch 解析的通用 LLM 依赖；`accelerate` 会用 `--no-deps` 安装，避免它自动拉取 PyPI 默认 torch。

如果已经误装 CUDA 版 PyTorch，运行：

```bash
bash scripts/setup_llm.sh --rocm
```

脚本会先卸载现有 `torch/torchvision/torchaudio`，并清理 `nvidia-*` CUDA wheel 残留依赖，再从 ROCm wheel 源安装 ROCm 版 PyTorch；如果最终 `torch.version.hip` 为空，脚本会直接报错停止。

安装后验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

AMD ROCm 正确结果：`torch.version.hip` 非空，且 `torch.cuda.is_available()` 为 `True`。如果 `torch.version.cuda` 非空但 `torch.version.hip` 为空，就是 CUDA 版 PyTorch，不能作为 AMD ROCm 实测。

### 2. 下载模型

```bash
bash scripts/download_llm.sh
```

模型将下载到 `models/qwen3-1.7b/` 目录（约 3.5GB）。首次下载需要较长时间。

如果下载失败：
- 检查网络连接
- 设置 Hugging Face 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 重新运行 `bash scripts/download_llm.sh`
- 或使用已有的 Hugging Face cache

### 3. 验证代码

```bash
python -m py_compile localdoc/backends/local_llm_backend.py
```

### 4. 测试模型

```bash
python scripts/test_llm.py
```

正常输出应包含：
- 模型加载信息
- 推理设备（CPU 或 GPU）
- 问题和回答
- 耗时和硬件说明

### 5. 启动 Demo

```bash
bash scripts/run_demo_llm.sh
```

访问 `http://localhost:7860`，上传文档后提问，系统将使用本地 Qwen3-1.7B 生成回答。

### 6. 运行 LLM Benchmark

```bash
bash scripts/run_llm_benchmark.sh
```

生成 LLM 生成延迟和 RAG 模式对比的 CSV 和图表。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCALDOC_USE_LLM` | `0` | 设为 `1` 启用 LLM 后端 |
| `LOCALDOC_LLM_MODEL_PATH` | `models/qwen3-1.7b` | 本地模型目录 |
| `LOCALDOC_LLM_MODEL_ID` | `Qwen/Qwen3-1.7B` | Hugging Face 模型 ID |
| `LOCALDOC_LLM_MAX_NEW_TOKENS` | `128` | 最大生成 token 数 |
| `LOCALDOC_LLM_CONTEXT_CHARS` | `1600` | 最大上下文字符数 |

## 如果生成过慢

把 `LOCALDOC_LLM_MAX_NEW_TOKENS` 改成 64：

```bash
export LOCALDOC_LLM_MAX_NEW_TOKENS=64
bash scripts/run_demo_llm.sh
```

## 如果机器内存不够

直接关闭 LLM 模式，回退到原始抽取式回答：

```bash
bash run_demo.sh
```

## 如果 LLM 无法使用

如果模型下载失败或依赖安装有问题，系统会自动回退到抽取式回答生成，不影响其他功能。

---

## 替代方案：在真实 AMD 硬件上运行

### 方案 A：LM Studio + ROCm GPU 加速

1. 安装 [LM Studio](https://lmstudio.ai/)
2. 下载 Qwen3-1.7B 或其他量化模型
3. LM Studio 自动使用 ROCm GPU 加速推理
4. 启动本地 API Server（默认 `http://localhost:1234`）
5. 通过 OpenAI 兼容 API 接入 LocalDoc Agent

优势：可视化界面、ROCm GPU 加速、int4/int8 量化支持。

### 方案 B：Lemonade 推理框架（AMD 官方）

1. 安装：`pip install lemonade-sdk`
2. AMD 官方适配模型列表：https://huggingface.co/amd
3. 通过 Lemonade Server API 暴露本地推理端点

优势：AMD 官方支持、内置 NPU 调度、统一 API。

### 方案 C：Ryzers Docker 容器

1. 克隆：`git clone https://github.com/AMDResearch/Ryzers.git`
2. 构建：`./build.sh llm/ollama`
3. 运行：`./run.sh llm/ollama`

优势：环境隔离、开箱即用、AMD 官方维护。
