# LocalDoc Agent

> **面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体**

![课程实验项目](https://img.shields.io/badge/异构计算-课程实验项目-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**LocalDoc Agent** 是一个面向 **AMD Ryzen AI MAX+** 平台优化的本地文档知识库智能体系统。
系统实现了完整的文档加载、智能切块、向量嵌入、语义检索与智能问答流程，
并针对 AMD 异构计算架构（CPU / GPU / NPU）设计了多后端调度策略，
旨在充分利用 AMD 锐龙 AI MAX+ 处理器的异构计算能力。

> **课程实验项目**：本项目对应《异构计算》课程"实验类方案三"——
> 面向 AMD 平台的本地知识库 Agent 设计与实现。

---

## 目录

- [项目简介](#项目简介)
- [课程任务对应关系](#课程任务对应关系)
- [系统架构](#系统架构)
- [CPU/GPU/NPU 分工表](#cpugpunpu-分工表)
- [安装方法](#安装方法)
- [一键运行方法](#一键运行方法)
- [实验复现方法](#实验复现方法)
- [当前限制说明](#当前限制说明)
- [文件树](#文件树)
- [后续：在真实 AMD 环境补实验数据](#后续在真实-amd-环境补实验数据)
- [依赖说明](#依赖说明)
- [许可证](#许可证)

---

## 项目简介

本项目旨在设计并实现一个**本地化文档知识库智能体（Local Knowledge-Base Agent）**，
核心特征如下：

- **全本地运行**：文档解析、向量检索、答案生成均在本地完成，无需联网，保护数据隐私。
- **异构计算优化**：针对 AMD Ryzen AI MAX+ 的 CPU + iGPU (RDNA 3.5) + NPU (XDNA) 三类计算单元，
  设计了分层调度策略，将计算密集型任务分配到最合适的硬件后端。
- **可扩展后端架构**：通过抽象后端接口，支持在真实 AMD 硬件与 CPU 模拟模式之间无缝切换。
- **端到端 Agent 流程**：支持从原始文档到智能问答的完整链路，包含文档加载、文本切块、
  向量嵌入、相似度检索、上下文构建与答案生成等环节。

---

## 课程任务对应关系

本项目对应**异构计算课程实验类方案三**的要求，具体映射如下：

| 方案要求 | 本项目对应实现 | 状态 |
|----------|----------------|------|
| 文档加载与解析 | `localdoc/loader.py` — 支持 MD/TXT/PDF | ✅ 已实现 |
| 文本切块策略 | `localdoc/chunker.py` — 段落+句子两层切分 | ✅ 已实现 |
| 向量嵌入 | `localdoc/embedding.py` — TF-IDF + 后端接口 | ✅ 已实现 |
| 语义检索 | `localdoc/retriever.py` — 余弦相似度 Top-K | ✅ 已实现 |
| 答案生成 | `localdoc/generator.py` — 抽取式 + 后端接口 | ✅ 已实现 |
| 异构调度 | `localdoc/scheduler.py` — CPU/GPU/NPU 分层调度 | ✅ 已实现 |
| 后端抽象 | `localdoc/backends/` — CPU/GPU/NPU/Simulated 四后端 | ✅ 已实现 |
| 性能基准测试 | `experiments/benchmark_latency.py` — 多后端对比 | ✅ 已实现 |
| Web UI 演示 | `localdoc/app.py` — Gradio 界面 | ✅ 已实现 |
| 实验报告 | `docs/experiment_report_draft.md` | ✅ 已完成 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户交互层 (Gradio UI)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ 文档上传  │  │ 知识库管理│  │ 问答输入  │  │ 性能监控面板  │    │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └──────┬───────┘    │
├────────┼─────────────┼────────────┼───────────────┼─────────────┤
│        ▼             ▼            ▼               ▼             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Agent 编排层 (Orchestrator)               │    │
│  │  Pipeline: Load → Chunk → Embed → Store → Retrieve → Gen │    │
│  └───────────────────────┬─────────────────────────────────┘    │
├──────────────────────────┼──────────────────────────────────────┤
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              异构调度器 (Heterogeneous Scheduler)          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │    │
│  │  │ 任务分类  │→│ 后端选择  │→│ 结果聚合  │               │    │
│  │  └──────────┘  └──────────┘  └──────────┘               │    │
│  └──────┬──────────────┬──────────────┬────────────────────┘    │
├─────────┼──────────────┼──────────────┼─────────────────────────┤
│         ▼              ▼              ▼                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │  CPU 后端   │ │  GPU 后端   │ │  NPU 后端   │                 │
│  │ (NumPy/Pty) │ │(ROCm/HIP)  │ │(XDNA/ONNX) │                 │
│  │  主控/逻辑  │ │ 矩阵运算加速 │ │ 推理加速    │                │
│  └────────────┘ └────────────┘ └────────────┘                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 数据层 (Data Layer)                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │    │
│  │  │ 文档存储  │  │ 向量索引  │  │ 结果缓存  │               │    │
│  │  └──────────┘  └──────────┘  └──────────┘               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 模块说明

| 模块 | 路径 | 职责 |
|------|------|------|
| Loader | `localdoc/loader.py` | 文档加载与解析（MD/TXT/PDF） |
| Chunker | `localdoc/chunker.py` | 文本切块（段落+句子两层切分 + 重叠窗口） |
| Embedding | `localdoc/embedding.py` | 文本向量化（TF-IDF，后端接口预留） |
| Retriever | `localdoc/retriever.py` | 语义检索（余弦相似度 Top-K） |
| Generator | `localdoc/generator.py` | 答案生成（抽取式 + 后端接口预留） |
| Agent | `localdoc/agent.py` | 主控编排（RAG 管线：加载→切块→嵌入→检索→生成） |
| Scheduler | `localdoc/scheduler.py` | 异构调度（任务分类 → 后端选择 → 执行跟踪） |
| Backends | `localdoc/backends/` | 计算后端（CPU / AMD GPU / AMD NPU / Simulated NPU） |
| Utils | `localdoc/utils/` | 工具模块（统一日志） |

---

## CPU/GPU/NPU 分工表

本系统根据任务计算特征，将不同计算环节分配到最合适的硬件后端：

| 计算环节 | 计算特征 | 首选后端 | 备选后端 | 说明 |
|----------|----------|----------|----------|------|
| 文档解析 | I/O 密集 + 字符串处理 | **CPU** | — | 串行 I/O，CPU 最优 |
| 文本切块 | 字符串处理 + 规则匹配 | **CPU** | — | 纯逻辑运算，CPU 最优 |
| TF-IDF 向量化 | 稀疏矩阵运算 | **CPU** | GPU | CPU 对稀疏运算高效 |
| 密集向量嵌入 | 矩阵乘法 (GEMM) | **GPU** | NPU | GPU 大规模并行 |
| 向量相似度计算 | 向量点积 + 归一化 | **GPU** | CPU | 批量并行计算 |
| 语义推理 | 神经网络推理 | **NPU** | GPU | NPU 低功耗高效推理 |
| 答案生成 | 文本生成 / 模板填充 | **CPU** | — | 逻辑密集，CPU 最优 |
| 批量索引构建 | 大规模并行计算 | **GPU** | CPU | GPU 并行优势明显 |

### 异构调度策略

```
任务到达 → 特征分析 → 后端选择 → 并行执行 → 结果合并
                │
                ├─ I/O 密集 → CPU
                ├─ 稀疏计算 → CPU
                ├─ 密集矩阵 → GPU
                ├─ 神经推理 → NPU
                └─ 混合任务 → CPU 主控 + GPU/NPU 协同
```

---

## 安装方法

### 环境要求

- **Python**: 3.8 或更高版本
- **操作系统**: Linux (推荐 Ubuntu 22.04+) / macOS / Windows
- **硬件** (可选): AMD Ryzen AI MAX+ 处理器（无此硬件时使用 CPU 回退模式）

### 安装步骤

```bash
# 1. 克隆项目（或解压到目标目录）
cd localdoc-agent-amd-ai

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或: venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 依赖说明

核心依赖（`requirements.txt`）：

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| `numpy` | >= 1.21 | 数值计算、向量运算 |
| `gradio` | >= 3.40 | Web UI 界面 |
| `psutil` | >= 5.9 (可选) | 系统资源监控 |
| `matplotlib` | >= 3.5 (可选) | 性能图表绘制 |
| `scikit-learn` | >= 1.0 (可选) | TF-IDF 向量化 |

---

## 一键运行方法

### 运行交互式 Demo

```bash
bash run_demo.sh
```

启动 Gradio Web UI，默认访问地址：`http://localhost:7860`

功能：
- 上传文档（PDF/TXT/MD）
- 自动构建知识库
- 输入问题，获取基于文档的回答
- 实时查看后端调度信息与性能指标

### 运行基准测试

```bash
bash run_benchmark.sh
```

自动执行以下测试：
- 各模块单独性能测试（加载、切块、嵌入、检索、生成）
- CPU vs GPU vs NPU 后端对比（无 GPU/NPU 时自动回退到 CPU 模拟）
- 端到端流程性能评估
- 结果输出到 `results/` 目录

---

## 实验复现方法

### 完整复现步骤

```bash
# 步骤 1：环境准备
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 步骤 2：运行基准测试
bash run_benchmark.sh

# 步骤 3：运行演示
bash run_demo.sh

# 步骤 4：查看实验结果
ls results/
# - benchmark_results.json    : 原始测试数据
# - performance_report.txt    : 可读性能报告
# - comparison_chart.png      : 性能对比图（需 matplotlib）

# 步骤 5：查看实验报告
cat docs/experiment_report_draft.md
```

### 注意事项

1. 本项目在非 AMD AI MAX+ 环境下，GPU/NPU 后端会自动回退到 CPU 模拟模式。
2. 所有实验结果均会标注实际使用的后端类型，不会伪造硬件测试数据。
3. 如需在真实 AMD 硬件上运行，请参考 `docs/amd_ai_max_backend.md` 配置指南。

---

## 当前限制说明

> **重要声明**

本项目当前版本存在以下限制：

1. **硬件依赖**：当前代码在开发环境中运行时，由于没有 AMD Ryzen AI MAX+ 硬件，
   GPU 后端和 NPU 后端均使用 **CPU 回退 + 模拟后端**。所有基准测试结果均基于 CPU 模拟模式，
   **不代表真实 AMD 硬件的性能表现**。

2. **不伪造数据**：本项目**不会**伪造 AMD 硬件的性能数据。所有输出均会明确标注
   实际使用的后端类型（`CPU` / `Simulated GPU` / `Simulated NPU`）。

3. **真实硬件补充**：如需获取真实 AMD 硬件性能数据，需要在配备 AMD Ryzen AI MAX+
   处理器的设备上运行基准测试，并参考 `docs/amd_ai_max_backend.md` 进行后端替换。

4. **LLM 集成**：当前答案生成模块使用模板抽取式生成，如需接入大语言模型，
   需额外配置 LLM 推理后端（如 ONNX Runtime AI 上的量化模型）。

5. **文档格式**：当前支持 PDF、TXT、Markdown 三种格式，其他格式（如 DOCX、HTML）
   需要扩展 Loader 模块。

---

## 文件树

```
localdoc-agent-amd-ai/
├── README.md                           # 项目说明文档
├── LICENSE                             # MIT 许可证
├── requirements.txt                    # Python 依赖清单
├── setup.py                            # 包安装配置
├── .gitignore                          # Git 忽略规则
├── run_demo.sh                         # 一键启动演示脚本
├── run_benchmark.sh                    # 一键运行基准测试脚本
│
├── localdoc/                           # 核心代码包
│   ├── __init__.py                     # 包初始化，版本号
│   ├── loader.py                       # 文档加载模块 (MD/TXT/PDF)
│   ├── chunker.py                      # 文本切块模块 (段落+句子两层切分)
│   ├── embedding.py                    # 向量嵌入模块 (TF-IDF + 后端接口)
│   ├── retriever.py                    # 语义检索模块 (余弦相似度 Top-K)
│   ├── generator.py                    # 答案生成模块 (抽取式 + 后端接口)
│   ├── agent.py                        # Agent 主控模块 (RAG 管线编排)
│   ├── scheduler.py                    # 异构调度器 (CPU/GPU/NPU 任务分配)
│   ├── app.py                          # Gradio Web UI 界面
│   ├── backends/                       # 计算后端模块
│   │   ├── __init__.py                 # 后端统一导出
│   │   ├── cpu_backend.py              # CPU 后端 (纯 Python TF-IDF)
│   │   ├── gpu_backend.py              # AMD GPU 后端 (ROCm/HIP)
│   │   ├── npu_backend.py              # AMD NPU 后端 (ONNX/Ryzen AI SDK)
│   │   └── simulated_npu.py            # 模拟 NPU 后端 (仅演示用)
│   └── utils/                          # 工具模块
│       ├── __init__.py
│       └── logger.py                   # 统一日志模块
│
├── experiments/                        # 实验脚本
│   ├── __init__.py
│   ├── benchmark_latency.py            # 延迟基准测试 (生成 CSV)
│   └── plot_results.py                 # 结果绘图脚本 (生成 PNG)
│
├── tests/                              # 单元测试 (28 个用例)
│   ├── __init__.py
│   ├── conftest.py                     # 共享 fixtures
│   ├── test_chunker.py                 # 文本切块测试
│   ├── test_retriever.py               # 语义检索测试
│   └── test_scheduler.py               # 异构调度器测试
│
├── results/                            # 实验结果 CSV
│   ├── latency_results.csv             # 延迟测试结果
│   ├── backend_results.csv             # 后端对比结果
│   └── resource_usage.csv              # 资源使用快照
│
├── figures/                            # 实验图表
│   ├── latency_comparison.png          # 延迟对比图
│   ├── backend_comparison.png          # 后端性能对比图
│   └── resource_usage.png              # 资源使用图
│
└── docs/                               # 文档目录
    ├── system_design.md                # 系统设计文档
    ├── reproduction.md                 # 实验复现指南
    ├── amd_ai_max_backend.md           # AMD 硬件后端替换指南
    └── experiment_report_draft.md      # 实验报告草稿 (中文)
```

---

## 后续：在真实 AMD 环境补实验数据

当获取到 AMD Ryzen AI MAX+ 硬件后，按以下步骤补充真实实验数据：

### 第一步：环境配置

```bash
# 安装 ROCm 6.x 驱动
sudo apt install rocm-hip-sdk rocm-hip-runtime

# 安装 Ryzen AI SDK
# 参考 docs/amd_ai_max_backend.md

# 安装 ONNX Runtime AI
pip install onnxruntime-directml  # 或 onnxruntime-gpu (ROCm)
```

### 第二步：替换后端

```bash
# 设置环境变量，启用真实后端
export LOCALDOC_BACKEND_GPU=rocm
export LOCALDOC_BACKEND_NPU=xdna
```

### 第三步：运行真实基准测试

```bash
bash run_benchmark.sh --real-hardware
```

### 第四步：更新实验报告

将 `docs/experiment_report_draft.md` 中的模拟数据替换为真实测试数据，
并在所有标注 `[模拟数据]` 的位置更新为 `[实测数据]`。

详细步骤请参考：[docs/amd_ai_max_backend.md](docs/amd_ai_max_backend.md)

---

## 许可证

本项目采用 MIT 许可证，仅供学术研究与课程实验使用。
