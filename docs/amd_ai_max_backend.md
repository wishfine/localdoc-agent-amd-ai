# LocalDoc Agent — AMD Ryzen AI MAX+ 后端配置指南

> 本文档指导如何在真实 AMD Ryzen AI MAX+ 硬件上配置和运行 LocalDoc Agent，
> 将 CPU 模拟后端替换为真实的 GPU/NPU 后端。

---

## 目录

1. [AMD Ryzen AI MAX+ 硬件概述](#1-amd-ryzen-ai-max-硬件概述)
2. [ROCm/HIP 环境配置](#2-rocmhip-环境配置)
3. [ONNX Runtime AI / Ryzen AI SDK 配置](#3-onnx-runtime-ai--ryzen-ai-sdk-配置)
4. [Lemonade 推理框架（推荐）](#4-lemonade-推理框架推荐)
5. [Ryzers Docker 容器环境](#5-ryzers-docker-容器环境)
6. [LM Studio + ROCm 替代方案](#6-lm-studio--rocm-替代方案)
7. [如何替换 GPU 后端](#7-如何替换-gpu-后端)
8. [如何替换 NPU 后端](#8-如何替换-npu-后端)
9. [验证步骤](#9-验证步骤)
10. [性能对比预期](#10-性能对比预期)

---

## 1. AMD Ryzen AI MAX+ 硬件概述

### 1.1 处理器架构

AMD Ryzen AI MAX+ 系列（代号 Strix Halo）是 AMD 面向高性能 AI PC 推出的 SoC 处理器，
集成了三类异构计算单元：

| 计算单元 | 架构 | 特点 | 适用任务 |
|----------|------|------|----------|
| **CPU** | Zen 5 | 最多 16 核 32 线程，通用计算 | 逻辑控制、I/O、串行任务 |
| **iGPU** | RDNA 3.5 | 最多 40 个 CU，高性能图形与计算 | 大规模矩阵运算、并行计算 |
| **NPU** | XDNA (Ryzen AI) | 专用 AI 加速器，50+ TOPS | 神经网络推理、Transformer 加速 |

### 1.2 关键技术特性

- **统一内存架构**：CPU、GPU、NPU 共享系统内存（最高 128GB LPDDR5X），
  无需显式数据拷贝，极大降低异构计算的数据传输开销。
- **ROCm 生态**：GPU 计算通过 ROCm (Radeon Open Compute) 平台实现，
  兼容 HIP (Heterogeneous-compute Interface for Portability) 编程模型。
- **Ryzen AI SDK**：NPU 通过 Ryzen AI SDK 进行编程，
  支持 ONNX Runtime AI 的 Vitis AI Execution Provider。

### 1.3 典型产品型号

| 型号 | CPU 核心 | GPU CU 数 | NPU TOPS | 内存 |
|------|----------|-----------|----------|------|
| Ryzen AI MAX+ 395 | 16C/32T | 40 CU | 50+ | 最高 128GB |
| Ryzen AI MAX 390 | 12C/24T | 32 CU | 50+ | 最高 128GB |
| Ryzen AI MAX 385 | 8C/16T | 32 CU | 50+ | 最高 128GB |

---

## 2. ROCm/HIP 环境配置

### 2.1 安装 ROCm

AMD GPU 的通用计算通过 ROCm 平台实现。

```bash
# === Ubuntu 22.04/24.04 ===

# 2.1.1 添加 ROCm 仓库
sudo apt update
sudo apt install -y wget gnupg2
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.3 jammy main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# 2.1.2 安装 ROCm SDK
sudo apt update
sudo apt install -y rocm-hip-sdk rocm-hip-runtime rocm-dev

# 2.1.3 配置用户权限
sudo usermod -aG video,render $USER
# 需要重新登录使权限生效

# 2.1.4 配置环境变量
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 2.1.5 验证安装
rocm-smi
# 应显示 GPU 设备信息

hipconfig --version
# 应显示 HIP 版本号
```

### 2.2 验证 GPU 计算能力

```bash
# 运行 HIP 示例程序
cd /opt/rocm/share/hip/samples
# 或编写简单测试:
python3 -c "
import subprocess
result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True)
print(result.stdout)
"
```

### 2.3 安装 CuPy（Python GPU 计算库）

```bash
# 安装 CuPy ROCm 版本
pip install cupy-rocm-6-0
# 或根据 ROCm 版本选择:
# pip install cupy-rocm-5-7  (ROCm 5.7)
# pip install cupy-rocm-6-0  (ROCm 6.0)

# 验证 CuPy
python3 -c "
import cupy as cp
print('CuPy version:', cp.__version__)
print('GPU device:', cp.cuda.runtime.getDeviceProperties(0)['name'])
a = cp.random.randn(1000, 1000)
b = cp.random.randn(1000, 1000)
c = cp.dot(a, b)
print('Matrix multiply test: OK, result shape:', c.shape)
"
```

### 2.4 Windows ROCm 配置（ROCm 7.12.0+）

ROCm 7.12.0 由 [TheRock](https://github.com/ROCm/TheRock) 构建，已支持 Windows 平台：

```bash
# Windows ROCm 文档：https://rocm.docs.amd.com/en/7.12.0-preview/

# 安装 ROCm Python SDK（Windows）
python -m pip install --no-cache-dir "https://repo.radeon.com/rocm/windows/.rocm-rel-7.2_a/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "https://repo.radeon.com/rocm/windows/.rocm-rel-7.2_a/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl"

# 安装 PyTorch ROCm 版（Windows）
python -m pip install --no-cache-dir "https://repo.radeon.com/rocm/windows/.rocm-rel-7.2_a/torch-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl"
python -m pip install --no-cache-dir "https://repo.radeon.com/rocm/windows/.rocm-rel-7.2_a/torchvision-0.24.1+rocmsdk20260116-cp312-cp312-win_amd64.whl"
```

> **注意**：Windows ROCm 版本号和 wheel URL 可能随版本更新而变化，
> 请参考 https://rocm.docs.amd.com/en/7.12.0-preview/ 获取最新链接。

### 2.5 Ryzen AI SW 环境管理注意事项

使用 Ryzen AI SDK 时，**不要直接修改基础环境**，应使用 conda 克隆：

```bash
# ✅ 正确做法：克隆环境后再开发
conda create --name localdoc_env --clone ryzen-ai-1.6.1
conda activate localdoc_env

# ❌ 错误做法：直接在 ryzen-ai-1.6.1 环境中修改 onnxruntime/quark 等库
# 这样会导致环境损坏，只能重新运行安装程序
```

### 2.6 Strix Halo (Ryzen AI MAX+) 统一内存优化

> **重要**：Ryzen AI MAX+ 采用统一内存架构，CPU 和 GPU 共享系统内存（最高 128GB LPDDR5X-8000）。
> 没有独立显存，需要专门的内存配置才能发挥最佳性能。

#### 2.6.1 BIOS 设置

**VRAM carve-out 设为最小值（如 0.5 GB）**。

原因：统一内存架构下，静态预留大块显存会永久减少可用系统内存，
而 GPU 实际使用的是动态 GTT 内存池，不需要大块静态预留。

BIOS 中相关选项名称可能为：
- VRAM / Carve-out / GART / Dedicated GPU memory / Firmware-reserved GPU memory

#### 2.6.2 配置 TTM 页面限制

TTM (Translation Table Manager) 控制 GPU 可使用的系统内存量：

```bash
# 安装 amd-ttm 工具
sudo apt install pipx
pipx ensurepath
pipx install amd-debug-tools

# 查看当前配置
amd-ttm

# 设置 GPU 可用内存（128GB 系统建议设 100GB）
amd-ttm --set 100

# 重启生效
sudo reboot
```

配置写入 `/etc/modprobe.d/ttm.conf`。

#### 2.6.3 内核版本硬性要求

Ryzen AI MAX+ (gfx1151) 需要特定内核补丁，低于以下版本会导致
GPU 计算工作负载初始化失败或行为异常：

| 发行版 | 最低内核版本 |
|--------|-------------|
| Ubuntu 24.04 HWE | `6.17.0-19.19~24.04.2` |
| Ubuntu 24.04 OEM | `6.14.0-1018` |
| 其他发行版 | Linux kernel `6.18.4` |
| Fedora 43 / Ubuntu 26.04 / Arch 2026.02 | 原生支持 |

#### 2.6.4 ROCm 版本兼容性（Strix Halo）

| ROCm 版本 | 兼容内核 | 旧内核 |
|-----------|---------|--------|
| 7.2.1 / 7.11.0+ | ✅ 稳定 | ⚠️ 不稳定 |
| 7.2.0 | ✅ 稳定（仅 HWE/OEM/6.18.4+） | ❌ 不支持 |
| 7.9.0 / 7.10.0 | ❌ 不支持 | ⚠️ 不稳定 |

#### 2.6.5 APU 框架限制

**Ryzen APU 上 ROCm 仅支持 PyTorch**，不支持 TensorFlow / JAX / ONNX Runtime。

| 框架 | Ryzen APU | 独立 Radeon GPU |
|------|-----------|----------------|
| PyTorch | ✅ | ✅ |
| TensorFlow | ❌ | ✅ |
| JAX | ❌ | ✅ |
| ONNX Runtime | ❌ | ✅ |

这意味着：
- `localdoc/backends/gpu_backend.py`（基于 PyTorch）✅ 是 APU 上的正确路径
- `localdoc/backends/npu_backend.py`（基于 ONNX Runtime）→ NPU 不走 ROCm，需通过 Ryzen AI SDK / Lemonade

#### 2.6.6 NPU 与 ROCm 的关系

**ROCm 不覆盖 NPU**。ROCm 文档中没有任何 NPU 相关内容。

NPU 推理需要通过以下方式：
- **Ryzen AI SDK**（ONNX Runtime + VitisAI EP）
- **Lemonade**（AMD 推理框架，推荐）
- **Ryzers Docker 容器**（预配置环境）

在本项目中：
- GPU 后端（`AMDGPUBackend`）→ 通过 ROCm + PyTorch
- NPU 后端（`AMDNPUBackend`）→ 通过 Ryzen AI SDK + ONNX Runtime（不走 ROCm）

---

## 3. ONNX Runtime AI / Ryzen AI SDK 配置

### 3.1 Ryzen AI SDK 概述

AMD Ryzen AI SDK 提供了在 XDNA NPU 上运行 AI 推理的能力。
它通过 ONNX Runtime 的 Vitis AI Execution Provider (EP) 来调度 NPU 计算。

### 3.2 安装 Ryzen AI SDK

```bash
# 3.2.1 安装 Ryzen AI 软件包
# 参考 AMD 官方文档: https://ryzenai.docs.amd.com/

# 方式一：通过 pip 安装（推荐）
pip install onnxruntime-vitisai

# 方式二：手动安装（获取最新版本）
# 从 AMD 官方 GitHub 下载:
# https://github.com/amd/RyzenAI-SW

# 3.2.2 安装 NPU 驱动
# NPU 驱动通常包含在 AMD 芯片组驱动中
# 确保使用最新版本的 AMD 芯片组驱动
sudo apt install -y amd-npu-driver  # 或从 AMD 官网手动安装

# 3.2.3 验证 NPU 设备
ls /dev/accel*
# 应显示 NPU 设备节点
```

### 3.3 配置 ONNX Runtime

```python
# 验证 ONNX Runtime Vitis AI EP
import onnxruntime as ort

# 查看可用的 EP 列表
print("Available execution providers:")
for ep in ort.get_available_providers():
    print(f"  - {ep}")

# 预期输出应包含:
#   - VitisAIExecutionProvider  (NPU)
#   - CPUExecutionProvider      (CPU 回退)
```

### 3.4 准备量化模型

NPU 推理需要使用经过量化的 ONNX 模型：

```bash
# 安装量化工具
pip install onnxruntime-tools

# 下载预训练的 sentence-transformers 模型并转换为 ONNX
pip install sentence-transformers onnx

python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

# 导出为 ONNX
dummy_input = tokenizer("test input", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "embedding_model.onnx",
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['embeddings'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'embeddings': {0: 'batch'}
    }
)
print("ONNX model exported.")
EOF

# 量化模型（INT8，适配 NPU）
python3 -m onnxruntime.quantization.quantize \
    --input embedding_model.onnx \
    --output embedding_model_quantized.onnx \
    --quant_format QDQ
```

---

## 4. Lemonade 推理框架（推荐）

### 4.1 简介

[Lemonade](https://github.com/lemonade-sdk/lemonade) 是 AMD 生态下的轻量级 AI 推理框架，
专注于**推理加速、后端管理与多设备调度**。它封装了 NPU 调度逻辑，提供统一的推理 API，
比直接使用裸 ONNX Runtime 更简单。

- 项目地址：https://github.com/lemonade-sdk/lemonade
- 文档地址：https://lemonade-server.ai/docs/server/
- AMD 官方适配模型：https://huggingface.co/amd

### 4.2 为什么推荐 Lemonade

| 对比项 | 裸 ONNX Runtime | Lemonade |
|--------|----------------|----------|
| NPU 调度 | 手动配置 EP | 自动调度 |
| 模型适配 | 需要手动量化 | AMD 官方预优化模型 |
| API 复杂度 | 较高 | 简单统一 |
| 多设备支持 | 需要自己实现 | 内置 CPU/GPU/NPU 切换 |

### 4.3 在真实 AMD 环境中使用 Lemonade

```bash
# 安装 Lemonade
pip install lemonade-sdk

# 下载 AMD 官方适配的模型（如 Qwen 系列）
# 参考 https://huggingface.co/amd 获取适配列表

# 通过 Lemonade Server API 暴露本地推理端点
lemonade serve --model <model-name>
```

在 LocalDoc Agent 中，可以通过 OpenAI 兼容 API 接入 Lemonade：
```python
# 设置环境变量指向 Lemonade Server
export OPENAI_API_BASE=http://localhost:8000/v1
```

---

## 5. Ryzers Docker 容器环境

### 5.1 简介

[AMDResearch/Ryzers](https://github.com/AMDResearch/Ryzers) 是 AMD Research 提供的
Docker 容器化工具链，专为 AMD Ryzen AI 硬件（iGPU、NPU）量身定制。
每个镜像最小化主机依赖，模块化设计支持跨应用复用。

- 项目地址：https://github.com/AMDResearch/Ryzers
- 维护方：AMD Research（官方）
- License：MIT

### 5.2 快速上手

```bash
# 环境要求：Ubuntu 22.04/24.04 + Docker

# 克隆仓库
git clone https://github.com/AMDResearch/Ryzers.git
cd Ryzers

# 构建目标镜像（以 LLM 推理为例）
./build.sh llm/ollama

# 运行容器（自动挂载 GPU 设备）
./run.sh llm/ollama
```

### 5.3 与本项目相关的 Ryzers 镜像

| 镜像类别 | 用途 | 与本项目关系 |
|----------|------|-------------|
| LLM（Ollama/Llamacpp） | 本地大模型推理 | 替代 LocalLLMBackend |
| NPU（XDNA/IRON） | NPU 加速工具链 | 替代 AMDNPUBackend |
| 视觉（OpenCV/SAM） | 目标检测 | 扩展文档 OCR |

主机仅需安装 amdgpu 内核驱动，容器内已预配 ROCm 驱动绑定。

---

## 6. LM Studio + ROCm 替代方案

### 6.1 简介

[LM Studio](https://lmstudio.ai/) 是一个本地 LLM 客户端，支持 ROCm GPU 加速。
它可以加载量化模型并提供 OpenAI 兼容 API，无需修改 LocalDoc Agent 核心代码。

### 6.2 使用方式

1. 安装 LM Studio：https://lmstudio.ai/
2. 下载 Qwen3-1.7B 或其他模型
3. 启动本地 API Server（默认 `http://localhost:1234`）
4. 在 LocalDoc Agent 中通过环境变量接入：

```bash
export OPENAI_API_BASE=http://localhost:1234/v1
export OPENAI_API_KEY=lm-studio
export LOCALDOC_USE_LLM=1
python localdoc/app.py
```

### 6.3 优势

- ROCm GPU 加速推理（int4/int8 量化）
- 可视化模型管理界面
- 不需要修改 LocalDoc Agent 代码
- 支持多种开源模型（Qwen、Mistral、Phi、Llama 等）

---

## 7. 如何替换 GPU 后端

### 4.1 修改后端实现文件

编辑 `localdoc/backends/gpu_backend.py`，将模拟实现替换为真实的 ROCm GPU 调用：

```python
# localdoc/backends/gpu_backend.py

import numpy as np
from .base_backend import BaseBackend
from localdoc.utils.logger import get_logger

logger = get_logger(__name__)

class AMDGPUBackend(BaseBackend):
    """AMD GPU 计算后端 (ROCm/HIP)"""

    def __init__(self):
        self._cupy = None
        self._available = False
        self._device_name = "Unknown"
        self._try_init()

    def _try_init(self):
        """尝试初始化 CuPy (ROCm)"""
        try:
            import cupy as cp
            self._cupy = cp
            props = cp.cuda.runtime.getDeviceProperties(0)
            self._device_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
            self._available = True
            logger.info(f"AMD GPU 后端初始化成功: {self._device_name}")
        except Exception as e:
            logger.warning(f"AMD GPU 后端不可用，将使用 CPU 回退: {e}")
            self._available = False

    @property
    def name(self) -> str:
        return f"AMD GPU ({self._device_name})"

    def is_available(self) -> bool:
        return self._available

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.matmul(a, b)
        cp = self._cupy
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        c_gpu = cp.dot(a_gpu, b_gpu)
        return cp.asnumpy(c_gpu)

    def cosine_similarity(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        if not self._available:
            # CPU 回退
            norms = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
            return np.dot(documents, query) / (norms + 1e-10)
        cp = self._cupy
        q_gpu = cp.asarray(query)
        d_gpu = cp.asarray(documents)
        norms = cp.linalg.norm(d_gpu, axis=1) * cp.linalg.norm(q_gpu)
        scores = cp.dot(d_gpu, q_gpu) / (norms + 1e-10)
        return cp.asnumpy(scores)

    def dot_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.dot(a, b)
        cp = self._cupy
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        return cp.asnumpy(cp.dot(a_gpu, b_gpu))
```

### 4.2 设置环境变量

```bash
# 启用真实 GPU 后端
export LOCALDOC_BACKEND_GPU=rocm

# 可选：指定 GPU 设备
export HIP_VISIBLE_DEVICES=0
```

### 4.3 回退机制

如果 GPU 后端初始化失败，系统会：
1. 记录警告日志
2. 自动回退到 CPU 后端
3. 在性能报告中标注实际使用的后端

---

## 8. 如何替换 NPU 后端

### 5.1 修改 NPU 后端实现

编辑 `localdoc/backends/npu_backend.py`：

```python
# localdoc/backends/npu_backend.py

import numpy as np
from .base_backend import BaseBackend
from localdoc.utils.logger import get_logger

logger = get_logger(__name__)

class AMDNPUBackend(BaseBackend):
    """AMD NPU 计算后端 (XDNA / ONNX Runtime Vitis AI EP)"""

    def __init__(self, model_path: str = "models/embedding_model_quantized.onnx"):
        self._session = None
        self._available = False
        self._model_path = model_path
        self._try_init()

    def _try_init(self):
        """尝试初始化 ONNX Runtime Vitis AI EP"""
        try:
            import onnxruntime as ort

            # 检查 VitisAIExecutionProvider 是否可用
            available_eps = ort.get_available_providers()
            if 'VitisAIExecutionProvider' not in available_eps:
                logger.warning("VitisAIExecutionProvider 不可用")
                self._available = False
                return

            # 创建推理会话
            self._session = ort.InferenceSession(
                self._model_path,
                providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
                provider_options=[{'config_file': 'vaip_config.json'}, {}]
            )
            self._available = True
            logger.info("AMD NPU 后端初始化成功 (VitisAIExecutionProvider)")
        except Exception as e:
            logger.warning(f"AMD NPU 后端不可用: {e}")
            self._available = False

    @property
    def name(self) -> str:
        return "AMD NPU (XDNA)"

    def is_available(self) -> bool:
        return self._available

    def infer(self, input_data: dict) -> np.ndarray:
        """运行 NPU 推理"""
        if not self._available:
            raise RuntimeError("NPU 后端未初始化")
        return self._session.run(None, input_data)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.matmul(a, b)
        # NPU 不直接做通用矩阵乘法，使用 CPU 回退
        return np.matmul(a, b)

    def cosine_similarity(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        if not self._available:
            norms = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
            return np.dot(documents, query) / (norms + 1e-10)
        # NPU 最适合做推理任务，向量相似度使用 CPU 或 GPU
        norms = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
        return np.dot(documents, query) / (norms + 1e-10)

    def embedding_infer(self, texts: list) -> np.ndarray:
        """使用 NPU 进行文本嵌入推理"""
        if not self._available:
            raise RuntimeError("NPU 后端未初始化")
        # 使用 ONNX 模型进行嵌入推理
        # 具体的 input 格式取决于所用模型
        inputs = self._preprocess(texts)
        return self._session.run(None, inputs)[0]
```

### 5.2 设置环境变量

```bash
# 启用真实 NPU 后端
export LOCALDOC_BACKEND_NPU=xdna

# 配置 Vitis AI
export VAIP_CONFIG_PATH=./vaip_config.json
```

### 5.3 创建 Vitis AI 配置文件

```json
{
    "vaip_config": {
        "version": "1.0",
        "model_dir": "./models/",
        "cache_dir": "./cache/",
        "cache_key": "localdoc_embedding"
    }
}
```

---

## 9. 验证步骤

### 6.1 硬件验证

```bash
# 检查 CPU
lscpu | grep "Model name"
# 预期: AMD Ryzen AI MAX+ 395 (或类似型号)

# 检查 GPU
rocm-smi --showproductname
# 预期: AMD Radeon Graphics (RDNA 3.5)

# 检查 NPU
ls /dev/accel*
# 预期: /dev/accel0 (或类似设备节点)
```

### 6.2 后端验证

```bash
python3 << 'EOF'
print("=" * 50)
print("LocalDoc Agent — 后端验证")
print("=" * 50)

# CPU 后端
from localdoc.backends.cpu_backend import CPUBackend
cpu = CPUBackend()
print(f"\n[CPU] 名称: {cpu.name}")
print(f"[CPU] 可用: {cpu.is_available()}")
import numpy as np
a = np.random.randn(100, 100)
b = np.random.randn(100, 100)
c = cpu.matmul(a, b)
print(f"[CPU] 矩阵乘法测试: OK (shape={c.shape})")

# GPU 后端
try:
    from localdoc.backends.gpu_backend import AMDGPUBackend
    gpu = AMDGPUBackend()
    print(f"\n[GPU] 名称: {gpu.name}")
    print(f"[GPU] 可用: {gpu.is_available()}")
    if gpu.is_available():
        c = gpu.matmul(a, b)
        print(f"[GPU] 矩阵乘法测试: OK (shape={c.shape})")
except Exception as e:
    print(f"\n[GPU] 错误: {e}")

# NPU 后端
try:
    from localdoc.backends.npu_backend import AMDNPUBackend
    npu = AMDNPUBackend()
    print(f"\n[NPU] 名称: {npu.name}")
    print(f"[NPU] 可用: {npu.is_available()}")
except Exception as e:
    print(f"\n[NPU] 错误: {e}")

# 模拟 NPU 后端
from localdoc.backends.simulated_npu import SimulatedNPUBackend
sim_npu = SimulatedNPUBackend()
print(f"\n[模拟NPU] 名称: {sim_npu.name}")
print(f"[模拟NPU] 可用: {sim_npu.is_available()}")

print("\n" + "=" * 50)
EOF
```

### 6.3 端到端验证

```bash
# 运行完整基准测试
bash run_benchmark.sh

# 检查结果中后端标识
python3 -c "
import json
with open('results/benchmark_results.json') as f:
    data = json.load(f)
print('后端测试结果:')
for backend, results in data.get('backend_tests', {}).items():
    status = results.get('status', 'unknown')
    print(f'  {backend}: {status}')
"
```

### 6.4 预期验证输出

成功配置后，预期输出如下：

```
[CPU] 名称: CPU (NumPy)
[CPU] 可用: True
[CPU] 矩阵乘法测试: OK (shape=(100, 100))

[GPU] 名称: AMD GPU (AMD Radeon Graphics)
[GPU] 可用: True
[GPU] 矩阵乘法测试: OK (shape=(100, 100))

[NPU] 名称: AMD NPU (XDNA)
[NPU] 可用: True
```

---

## 10. 性能对比预期

### 7.1 预期性能提升

以下是基于 AMD Ryzen AI MAX+ 平台的预期性能对比（相对于纯 CPU 基准）：

| 计算环节 | CPU | GPU (RDNA 3.5) | NPU (XDNA) | 说明 |
|----------|-----|----------------|------------|------|
| 密集矩阵乘法 (1000x1000) | 1x (基准) | 5-15x | N/A | GPU 并行加速 |
| 大规模余弦相似度 (10000向量) | 1x | 8-20x | N/A | GPU 批量计算 |
| Sentence-BERT 推理 | 1x | 3-8x | 10-30x | NPU 低功耗高效推理 |
| TF-IDF 向量化 | 1x | 0.8-1.2x | N/A | CPU 已足够高效 |
| 文档加载/切块 | 1x | N/A | N/A | I/O 密集，CPU 最优 |

> **注意**：以上数据为理论预期值，实际性能取决于模型复杂度、数据规模、
> 内存带宽等因素。真实数据需在实际硬件上测量。

### 7.2 功耗对比

| 后端 | 典型功耗 | 性能/瓦特 |
|------|----------|-----------|
| CPU (全核) | 45-65W | 中等 |
| GPU (满载) | 50-120W | 高（大规模计算） |
| NPU (推理) | 5-15W | 最高（AI 推理任务） |

### 7.3 调度策略建议

综合性能与功耗考虑，推荐的调度策略：

```
I/O 密集任务    → CPU（零传输开销）
稀疏矩阵计算    → CPU（内存局部性好）
小规模矩阵运算  → CPU（传输开销大于计算收益）
大规模矩阵运算  → GPU（并行加速明显）
神经网络推理    → NPU（性能/瓦特最优）
混合任务        → CPU 主控 + GPU/NPU 协同
```
