# 本地 LLM 接入指南

## 为什么选择 Qwen3-1.7B

| 特征 | 说明 |
|------|------|
| 参数量 | 1.7B，轻量级，适合临时实验环境 |
| 语言支持 | 中文、英文 |
| 许可证 | Apache-2.0（Hugging Face） |
| 依赖 | 仅需 transformers + torch，不需要 vLLM/Ollama |
| 用途 | 课程展示本地 LLM 推理能力 |

## 重要声明

- 该 LLM 是**可选后端**，不影响默认的 CPU fallback 抽取式回答。
- 该 LLM **完全本地运行**，不调用任何云端 API。
- 如果没有真实 AMD GPU/NPU 硬件，模型在 **CPU 上推理**，不代表 GPU/NPU 实测。
- 该实验只证明"本地 LLM 生成链路可运行"，不声称 AMD GPU/NPU 加速。

## 安装步骤

### 1. 安装依赖

```bash
bash scripts/setup_llm.sh
```

这会安装 `requirements-llm.txt` 中的依赖（torch, transformers, accelerate 等）。

### 2. 下载模型

```bash
bash scripts/download_llm.sh
```

模型将下载到 `models/qwen3-1.7b/` 目录（约 3.5GB）。

如果下载失败：
- 检查网络连接
- 设置 Hugging Face 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 重新运行 `bash scripts/download_llm.sh`
- 或使用已有的 Hugging Face cache

### 3. 测试模型

```bash
python scripts/test_llm.py
```

正常输出应包含：
- 模型加载信息
- 问题和回答
- 耗时和设备信息

### 4. 启动 Demo

```bash
bash scripts/run_demo_llm.sh
```

访问 `http://localhost:7860`，上传文档后提问，系统将使用本地 LLM 生成回答。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCALDOC_USE_LLM` | `0` | 设为 `1` 启用 LLM 后端 |
| `LOCALDOC_LLM_MODEL_PATH` | `models/qwen3-1.7b` | 本地模型目录 |
| `LOCALDOC_LLM_MODEL_ID` | `Qwen/Qwen3-1.7B` | Hugging Face 模型 ID |
| `LOCALDOC_LLM_MAX_NEW_TOKENS` | `256` | 最大生成 token 数 |
| `LOCALDOC_LLM_CONTEXT_CHARS` | `2000` | 最大上下文字符数 |

## 如果 LLM 无法使用

如果模型下载失败或依赖安装有问题，系统会自动回退到抽取式回答生成，不影响其他功能。

运行默认 Demo：
```bash
bash run_demo.sh
```
