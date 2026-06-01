# LocalDoc Agent — 实验复现指南

> 本文档提供详细的实验复现步骤，确保他人可以在相同环境下重现本项目的实验结果。

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装步骤](#2-安装步骤)
3. [运行 Demo](#3-运行-demo)
4. [运行基准测试](#4-运行基准测试)
5. [查看结果](#5-查看结果)
6. [故障排除](#6-故障排除)

---

## 1. 环境要求

### 1.1 硬件要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 x86_64 处理器 | AMD Ryzen AI MAX+ 395 |
| 内存 | 8 GB | 32 GB+ |
| 磁盘 | 500 MB 可用空间 | 2 GB+ |
| GPU | 无（CPU 回退模式） | AMD Radeon iGPU (RDNA 3.5) |
| NPU | 无（模拟模式） | AMD XDNA NPU |

> **重要说明**：本项目在没有 AMD GPU/NPU 硬件的环境下可以正常运行，
> 所有计算将回退到 CPU 模拟模式。实验结果将明确标注实际使用的后端类型。

### 1.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.8 或更高 | 推荐 3.10+ |
| pip | 21.0+ | Python 包管理器 |
| 操作系统 | Linux / macOS / Windows | 推荐 Ubuntu 22.04 LTS |
| Git | 2.30+ (可选) | 用于克隆代码仓库 |

### 1.3 Python 依赖

核心依赖（安装时自动配置）：

```
numpy>=1.21
gradio>=3.40
```

可选依赖：

```
psutil>=5.9        # 系统资源监控（基准测试中显示 CPU/内存使用率）
matplotlib>=3.5    # 生成性能对比图表
scikit-learn>=1.0  # TF-IDF 向量化的 sklearn 实现（作为对比）
```

---

## 2. 安装步骤

### 2.1 获取代码

```bash
# 方式一：从压缩包解压
tar -xzf localdoc-agent-amd-ai.tar.gz
cd localdoc-agent-amd-ai

# 方式二：从 Git 仓库克隆（如果适用）
git clone <repository_url>
cd localdoc-agent-amd-ai
```

### 2.2 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux / macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2.3 安装依赖

```bash
# 升级 pip（推荐）
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

### 2.4 验证安装

```bash
# 检查 Python 版本
python --version
# 应输出 Python 3.8.x 或更高版本

# 检查核心依赖
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import gradio; print('gradio:', gradio.__version__)"

# 检查项目模块
python -c "import localdoc; print('localdoc version:', localdoc.__version__)"
# 应输出: localdoc version: 0.1.0
```

如果所有命令均无报错，说明安装成功。

---

## 3. 运行 Demo

### 3.1 启动交互式演示

```bash
bash run_demo.sh
```

启动成功后，终端将显示：

```
Running on local URL: http://localhost:7860
```

### 3.2 使用 Web UI

1. 在浏览器中打开 `http://localhost:7860`

2. **上传文档**：
   - 在"文档上传"区域选择 PDF、TXT 或 Markdown 文件
   - 可同时上传多个文件
   - 点击"构建知识库"按钮

3. **开始问答**：
   - 在"问题输入"框中输入您的问题
   - 点击"提交"按钮
   - 系统将返回基于文档内容的回答

4. **查看系统信息**：
   - 右侧面板显示当前使用的后端类型
   - 显示各环节的耗时统计
   - 显示检索到的文本块与相似度分数

### 3.3 命令行运行

如果不希望使用 Web UI，可以通过 Python 脚本直接调用：

```python
from localdoc.loader import PDFLoader
from localdoc.chunker import SlidingWindowChunker
from localdoc.embedding import TFIDFEmbedder
from localdoc.retriever import CosineRetriever
from localdoc.generator import TemplateGenerator

# 加载文档
loader = PDFLoader()
documents = loader.load("your_document.pdf")

# 切块
chunker = SlidingWindowChunker(chunk_size=512, overlap=64)
chunks = chunker.chunk(documents)

# 嵌入
embedder = TFIDFEmbedder()
embeddings = embedder.embed(chunks)

# 构建索引
retriever = CosineRetriever()
retriever.build_index(chunks, embeddings)

# 检索
query = "您的问题"
query_embedding = embedder.embed_query(query)
results = retriever.retrieve(query_embedding, top_k=5)

# 生成答案
generator = TemplateGenerator()
answer = generator.generate(query, results)
print(answer)
```

---

## 4. 运行基准测试

### 4.1 执行基准测试

```bash
bash run_benchmark.sh
```

基准测试脚本将自动执行以下测试项目：

1. **模块单独测试**：
   - Loader 加载性能（不同格式、不同文件大小）
   - Chunker 切块性能（不同参数配置）
   - Embedder 嵌入性能（不同文档规模）
   - Retriever 检索性能（不同索引大小）
   - Generator 生成性能

2. **后端对比测试**：
   - CPU 后端基准测试
   - GPU 后端测试（如不可用则跳过并记录）
   - NPU 后端测试（如不可用则跳过并记录）

3. **端到端流程测试**：
   - 小规模文档（~1KB）
   - 中等规模文档（~100KB）
   - 大规模文档（~1MB）

### 4.2 测试参数配置

可通过环境变量调整测试参数：

```bash
# 设置测试文档大小（KB）
export BENCHMARK_DOC_SIZE=100

# 设置重复测试次数（取平均值）
export BENCHMARK_REPEAT=5

# 设置切块大小
export CHUNK_SIZE=512

# 设置检索 Top-K
export TOP_K=5
```

### 4.3 预期运行时间

| 测试项目 | 预期耗时 | 说明 |
|----------|----------|------|
| 模块单独测试 | 30-60 秒 | 快速单元级测试 |
| 后端对比测试 | 1-3 分钟 | 多后端重复测试 |
| 端到端流程测试 | 2-5 分钟 | 含文档加载与处理 |
| 全部测试 | 5-10 分钟 | 取决于机器性能 |

---

## 5. 查看结果

### 5.1 结果文件说明

基准测试完成后，结果保存在 `results/` 目录下：

```
results/
├── benchmark_results.json      # 原始测试数据（JSON 格式）
├── performance_report.txt      # 可读性能报告
├── comparison_chart.png        # 性能对比图（需 matplotlib）
└── system_info.json            # 测试环境信息
```

### 5.2 查看性能报告

```bash
# 查看可读报告
cat results/performance_report.txt
```

报告内容包括：
- 测试环境信息（CPU 型号、内存大小、操作系统等）
- 各模块性能指标（平均耗时、标准差、吞吐量）
- 后端对比结果（CPU / GPU / NPU 各项指标）
- 端到端流程耗时分析

### 5.3 查看原始数据

```bash
# 查看 JSON 原始数据
python -m json.tool results/benchmark_results.json
```

JSON 数据结构：

```json
{
  "metadata": {
    "timestamp": "2026-06-01T12:00:00",
    "python_version": "3.10.12",
    "platform": "Linux-5.15.0-x86_64"
  },
  "module_tests": {
    "loader": {
      "pdf": {"avg_ms": 150.3, "std_ms": 12.1},
      "txt": {"avg_ms": 5.2, "std_ms": 0.8}
    },
    "chunker": {
      "sliding_window": {"avg_ms": 23.5, "std_ms": 2.1}
    }
  },
  "backend_tests": {
    "cpu": {"matmul_1000": {"avg_ms": 45.2}, "cosine_1000": {"avg_ms": 12.3}},
    "gpu": {"status": "unavailable"},
    "npu": {"status": "simulated"}
  }
}
```

### 5.4 查看性能图表

如果安装了 matplotlib，可以查看可视化对比图：

```bash
# 图表已自动生成
open results/comparison_chart.png  # macOS
# 或: xdg-open results/comparison_chart.png  # Linux
```

---

## 6. 故障排除

### 6.1 常见问题

#### 问题 1：`ModuleNotFoundError: No module named 'localdoc'`

**原因**：未在项目根目录运行，或虚拟环境未激活。

**解决方案**：

```bash
# 确保在项目根目录
cd localdoc-agent-amd-ai

# 确保虚拟环境已激活
source venv/bin/activate

# 重新安装
pip install -r requirements.txt
```

#### 问题 2：`ImportError: cannot import name 'xxx' from 'localdoc.backends'`

**原因**：后端模块的依赖未安装（如 ROCm、ONNX Runtime）。

**解决方案**：

```bash
# 检查后端可用性
python -c "
from localdoc.backends import CPUBackend, AMDGPUBackend, AMDNPUBackend, SimulatedNPUBackend
for cls in [CPUBackend, AMDGPUBackend, AMDNPUBackend, SimulatedNPUBackend]:
    try:
        b = cls()
        print(f'{b.name}: available={b.is_available()}')
    except Exception as e:
        print(f'{cls.__name__}: error={e}')
"
```

如果 GPU/NPU 后端不可用，系统将自动使用 CPU 回退模式。

#### 问题 3：Gradio 启动失败

**原因**：端口被占用，或 Gradio 版本不兼容。

**解决方案**：

```bash
# 更换端口
export GRADIO_SERVER_PORT=7861
bash run_demo.sh

# 或升级 Gradio
pip install --upgrade gradio
```

#### 问题 4：PDF 加载失败

**原因**：缺少 PDF 解析库。

**解决方案**：

```bash
pip install PyPDF2
# 或
pip install pdfplumber
```

#### 问题 5：中文文本切块效果不佳

**原因**：缺少中文分词工具。

**解决方案**：

```bash
pip install jieba
```

### 6.2 性能调优

#### 内存不足

```bash
# 减小切块大小
export CHUNK_SIZE=256

# 减少测试规模
export BENCHMARK_DOC_SIZE=50
```

#### 测试太慢

```bash
# 减少重复次数
export BENCHMARK_REPEAT=3

# 只运行部分测试
python -m experiments.benchmark --test-only loader,chunker
```

### 6.3 获取帮助

如遇到其他问题：

1. 检查 `results/system_info.json` 确认环境信息
2. 查看详细日志：`export LOCALDOC_LOG_LEVEL=DEBUG`
3. 参考 `docs/system_design.md` 了解系统设计细节
4. 参考 `docs/amd_ai_max_backend.md` 了解后端配置

---

## 附录：快速验证脚本

将以下内容保存为 `verify_install.py` 并运行，可快速验证安装是否成功：

```python
#!/usr/bin/env python3
"""快速验证 LocalDoc Agent 安装"""
import sys

def check(name, func):
    try:
        result = func()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("LocalDoc Agent 安装验证")
print("=" * 40)

print("\n1. Python 环境:")
check("Python 版本", lambda: sys.version)

print("\n2. 核心依赖:")
check("numpy", lambda: __import__("numpy").__version__)
check("gradio", lambda: __import__("gradio").__version__)

print("\n3. 可选依赖:")
check("psutil", lambda: __import__("psutil").__version__)
check("matplotlib", lambda: __import__("matplotlib").__version__)

print("\n4. 项目模块:")
check("localdoc", lambda: __import__("localdoc").__version__)
check("localdoc.backends.CPUBackend",
      lambda: __import__("localdoc.backends", fromlist=["CPUBackend"]).CPUBackend().name)

print("\n验证完成!")
```

```bash
python verify_install.py
```
