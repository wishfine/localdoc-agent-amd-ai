# 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体设计与异构资源调度仿真实验

![课程实验项目](https://img.shields.io/badge/异构计算-课程实验项目-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 项目定位

这是一个面向 AMD Ryzen AI MAX+ 平台设计的本地知识库智能体原型与异构资源调度仿真实验。项目实现了完整的 RAG（Retrieval-Augmented Generation）管线——文档加载、智能切块、向量嵌入、语义检索、答案生成——并通过抽象化的后端接口对 CPU / GPU / NPU 三种异构计算资源进行了统一调度。**当前仓库默认不包含真实 AMD AI MAX+ 硬件实测结果。没有 AMD 环境时，项目运行在 CPU fallback + simulated backend 模式**，所有 GPU/NPU 性能数据均为仿真产生，已在输出文件中明确标注。

---

## 实验诚信声明

> **本项目未伪造任何 AMD GPU/NPU 硬件实验结果。**
>
> 当前开发与测试环境中没有真实 AMD Ryzen AI MAX+ 硬件，所有 GPU/NPU 相关数据均基于 **CPU fallback + simulated backend** 产生，已在所有输出文件（CSV、图表、报告）中明确标注。若后续获得真实硬件，将以 `[实测数据]` 标注替换 `[模拟数据]` 标注。

---

## 课程要求对应关系

| 课程要求 | 本项目对应实现 | 说明 |
|----------|----------------|------|
| 本地 AI 推理 | 文档问答全流程本地完成，无需联网 | 文档解析、向量检索、答案生成均在本地执行 |
| 端到端应用 | 上传文档 -> 切块 -> 嵌入 -> 检索 -> 回答 -> 资源调度展示 | 完整 RAG Pipeline + Gradio UI |
| 异构资源分工 | CPU / GPU / NPU 后端抽象与调度策略 | CPU 实际执行；GPU/NPU 接口预留，需真实硬件激活 |
| 性能与能效 | 当前为 simulated latency benchmark | 使用模拟延迟，真实硬件可替换后端后补测 |

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
│  │  Pipeline: Load -> Chunk -> Embed -> Store -> Retrieve -> Gen││
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
│  │ (NumPy/Py)  │ │(ROCm/HIP)  │ │(XDNA/ONNX) │                 │
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

---

## 后端说明

| Backend | 当前状态 | 含义 |
|---------|----------|------|
| CPUBackend | 真实，始终可用 | 真实 CPU 执行，本项目默认后端 |
| AMDGPUBackend | 可选，需 ROCm PyTorch | 仅当调度器检测到 ROCm 环境时才使用真实后端 |
| AMDNPUBackend | 可选，需 Ryzen AI SDK / ONNX EP | 仅当调度器检测到 Ryzen AI SDK 时才使用真实后端 |
| SimulatedNPUBackend | 仅仿真 | 仅用于演示，不产生真实硬件结果 |

---

## 安装方法

### 环境要求

- **Python**: >= 3.9
- **操作系统**: Linux (推荐 Ubuntu 22.04+) / macOS / Windows
- **硬件** (可选): AMD Ryzen AI MAX+ 处理器（无此硬件时自动使用 CPU fallback 模式）

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

---

## 快速开始

### 运行交互式 Demo

```bash
bash run_demo.sh
```

启动 Gradio Web UI，默认访问地址：`http://localhost:7860`。可上传文档、构建知识库、输入问题获取回答、查看后端调度信息。

### 运行基准测试

```bash
bash run_benchmark.sh
```

自动执行各模块性能测试与多后端对比，结果输出到 `results/` 目录。在无 AMD 硬件的环境下，所有数据均为模拟结果。

### 运行测试

```bash
python -m pytest tests/ -v
```

运行全部单元测试用例，验证核心模块功能正确性。

---

## 输出文件说明

| 文件路径 | 说明 |
|----------|------|
| `results/environment_report.txt` | 环境检测报告（硬件、驱动、后端可用性） |
| `results/latency_results.csv` | 延迟基准测试结果（当前全部为模拟数据） |
| `results/backend_results.csv` | 多后端对比结果（当前全部为模拟数据） |
| `results/resource_usage.csv` | 系统资源使用快照 |
| `figures/*.png` | 性能对比图表 |

---

## 文件树

```
localdoc-agent-amd-ai/
├── README.md                           # 项目说明文档（本文件）
├── LICENSE                             # MIT 许可证
├── requirements.txt                    # Python 依赖清单
├── setup.py                            # 包安装配置
├── .gitignore                          # Git 忽略规则
├── run_demo.sh                         # 一键启动演示脚本
├── run_benchmark.sh                    # 一键运行基准测试脚本
├── ppt_outline.md                      # 答辩 PPT 大纲
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
│   │   ├── cpu_backend.py              # CPU 后端 (真实执行)
│   │   ├── gpu_backend.py              # AMD GPU 后端 (ROCm/HIP)
│   │   ├── npu_backend.py              # AMD NPU 后端 (ONNX/Ryzen AI SDK)
│   │   └── simulated_npu.py            # 模拟 NPU 后端 (仅演示用)
│   └── utils/                          # 工具模块
│       ├── __init__.py
│       └── logger.py                   # 统一日志模块
│
├── experiments/                        # 实验脚本
│   ├── __init__.py
│   ├── check_environment.py            # 环境检测脚本（硬件/驱动/后端可用性）
│   ├── benchmark_latency.py            # 延迟基准测试 (生成 CSV)
│   └── plot_results.py                 # 结果绘图脚本 (生成 PNG)
│
├── tests/                              # 单元测试
│   ├── __init__.py
│   ├── conftest.py                     # 共享 fixtures
│   ├── test_chunker.py                 # 文本切块测试
│   ├── test_retriever.py               # 语义检索测试
│   └── test_scheduler.py               # 异构调度器测试
│
├── results/                            # 实验结果 CSV
│   ├── environment_report.txt          # 环境检测报告
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

## 当前限制说明

1. **无真实 AMD 硬件**：当前开发环境中没有 AMD Ryzen AI MAX+ 处理器，GPU 后端和 NPU 后端均使用 CPU fallback + simulated backend。所有 GPU/NPU 性能数据均为仿真结果，不代表真实硬件表现。

2. **Simulated backend 数据**：`results/` 目录下的延迟数据和后端对比数据均为 simulated backend 产出。相关 CSV 文件和图表中已标注 "simulated" 字样。

3. **TF-IDF 嵌入（非神经网络）**：当前向量嵌入模块使用 TF-IDF，而非神经网络嵌入模型。在真实 AMD GPU/NPU 环境下可替换为 Dense Embedding 模型以利用硬件加速。

4. **模板式答案生成**：当前答案生成模块采用抽取式/模板式策略，未接入大语言模型。在真实硬件平台上可接入 Qwen3.5 4B 等模型实现神经网络生成。

5. **文档格式有限**：当前支持 PDF、TXT、Markdown 三种格式，其他格式（如 DOCX、HTML）需扩展 Loader 模块。

---

## 后续：在真实 AMD 环境补实验数据

当获取到 AMD Ryzen AI MAX+ 硬件后，按以下步骤补充真实实验数据：

### 第一步：运行环境检测

```bash
python experiments/check_environment.py
```

检测硬件、驱动、ROCm 版本、Ryzen AI SDK 可用性，生成 `results/environment_report.txt`。

### 第二步：GPU 后端自动激活

如果 ROCm 环境检测通过（`torch.version.hip` 可用），GPU 后端将在调度器中自动激活，无需手动配置。

### 第三步：NPU 后端自动激活

如果 Ryzen AI SDK / ONNX Execution Provider 检测通过，NPU 后端将在调度器中自动激活。

### 第四步：运行真实基准测试

```bash
bash run_benchmark.sh
```

此时 `results/` 目录下的数据将包含真实硬件性能数据。

### 第五步：更新实验报告

将 `docs/experiment_report_draft.md` 中所有 `[模拟数据]` 标注替换为 `[实测数据]`，并更新对应数据表格与图表。

### 第六步：接入 Qwen3.5 4B 模型（可选）

在真实 AMD 平台上，可通过 ONNX Runtime 将 Qwen3.5 4B 模型部署到 NPU 上运行，实现神经网络级别的答案生成，替代当前的模板式生成策略。

---

## 可选：接入本地小语言模型

默认版本不加载 LLM，保证普通 CPU 环境可运行。如果需要展示**本地 AI 推理**能力，可以启用 Qwen2.5-0.5B-Instruct 作为生成后端。

### 启用步骤

```bash
# 1. 安装 LLM 依赖（torch, transformers 等）
bash scripts/setup_llm.sh

# 2. 下载模型（约 1GB，从 Hugging Face）
bash scripts/download_llm.sh

# 3. 测试模型
python scripts/test_llm.py

# 4. 启动带 LLM 的 Demo
bash scripts/run_demo_llm.sh
```

### 说明

- 该 LLM **完全本地运行**，不调用任何云端 API。
- Embedding 仍使用 TF-IDF（CPUBackend），LLM 仅替换答案生成环节。
- 无真实 AMD 硬件时，模型在 **CPU 上推理**，不是 GPU/NPU 实测。
- 模型可从 `Qwen/Qwen2.5-0.5B-Instruct`（Hugging Face）下载，Apache-2.0 许可证。
- 详细说明见 [docs/local_llm_setup.md](docs/local_llm_setup.md)。

---

## 许可证

本项目采用 [MIT License](LICENSE)，仅供学术研究与课程实验使用。
