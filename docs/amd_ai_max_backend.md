# LocalDoc Agent — AMD Ryzen AI MAX+ 后端配置指南

> 本文档指导如何在真实 AMD Ryzen AI MAX+ 硬件上配置和运行 LocalDoc Agent，
> 将 CPU 模拟后端替换为真实的 GPU/NPU 后端。

---

## 目录

1. [AMD Ryzen AI MAX+ 硬件概述](#1-amd-ryzen-ai-max-硬件概述)
2. [ROCm/HIP 环境配置](#2-rocmhip-环境配置)
3. [ONNX Runtime AI / Ryzen AI SDK 配置](#3-onnx-runtime-ai--ryzen-ai-sdk-配置)
4. [如何替换 GPU 后端](#4-如何替换-gpu-后端)
5. [如何替换 NPU 后端](#5-如何替换-npu-后端)
6. [验证步骤](#6-验证步骤)
7. [性能对比预期](#7-性能对比预期)

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

## 4. 如何替换 GPU 后端

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

## 5. 如何替换 NPU 后端

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

## 6. 验证步骤

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

## 7. 性能对比预期

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
