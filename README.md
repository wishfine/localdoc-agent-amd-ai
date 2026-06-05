# 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体设计与异构资源调度仿真实验

![课程实验项目](https://img.shields.io/badge/异构计算-课程实验项目-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 项目定位

这是一个面向 AMD Ryzen AI MAX+ 平台设计的本地知识库智能体原型与异构资源调度实验项目。项目实现了完整的 RAG（Retrieval-Augmented Generation）管线：文档加载、智能切块、向量嵌入、语义检索、答案生成，并通过统一后端接口和调度器组织 CPU / GPU / NPU 三类异构资源。

当前代码已经支持在普通 CPU 环境、无 sudo 的 Jupyter 容器环境、以及 AMD ROCm Linux 环境中运行。实验脚本会自动区分真实 CPU、真实 ROCm GPU、CPU fallback、SimulatedNPU 和 unavailable 后端，并在 CSV 中用 `measurement_type`、`is_simulated`、`real_inference` 明确标注，避免把模拟数据写成真实硬件数据。

---

## 实验诚信声明

> **本项目不伪造 AMD GPU/NPU 硬件实验结果。**
>
> 已在 AMD/Jupyter 环境中检测到 ROCm 命令行工具和 `gfx1151` 架构，但当前运行记录显示 Python 环境尚未安装 ROCm 版 PyTorch，因此 ROCm GPU benchmark 仍未激活。此时基础实验只有 CPU baseline，Agent 的 SimulatedNPU 结果仍是 CPU 计算 + 人为延迟，不代表真实 AMD NPU/GPU 性能。
>
> **依赖修正说明**：旧版 `requirements-llm.txt` 曾直接写入 `torch>=2.2`。在 Linux 上执行普通 `pip install torch` / `pip install -r requirements-llm.txt` 可能安装成 CUDA 版 PyTorch，这不适用于 AMD ROCm 实测。当前已把 `torch` 从通用依赖中移除，并改为通过 `bash scripts/setup_llm.sh --rocm` 或 `--cpu` 显式选择平台。

---

## 最新运行状态

来自 AMD/Jupyter 环境的最新运行记录：

| 项目 | 状态 |
|------|------|
| 操作系统/内核 | Linux `6.14.0-1018-oem`，32 CPU 线程，约 64 GB 内存 |
| ROCm 命令行工具 | `rocminfo`、`rocm-smi`、`hipcc`、`hipconfig` 均可执行 |
| GPU 架构证据 | `gfx11`、`gfx1151` |
| PyTorch-HIP | 初始记录为未安装：`torch 已安装: False`；如果按旧依赖安装过，可能变成 CUDA 版 PyTorch，需要按下文清理重装 |
| ROCm GPU benchmark | 未激活：`ROCm GPU available: False` |
| 当前可作为报告证据的数据 | ROCm 环境证据、CPU baseline、完整端到端流程、模拟后端标注、能耗/资源采样 |
| 当前不能声称的数据 | 不能声称已经获得 ROCm GPU / Ryzen AI NPU 真实加速比 |

要获得真正的 `ROCm_GPU` 实测行，需要在该 AMD 环境中安装匹配当前 ROCm 版本的 PyTorch-HIP，然后确认：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

只有当 `torch.version.hip` 非空且 `torch.cuda.is_available()` 为 `True` 时，`experiments/basic_benchmarks.py` 才会自动写入 `ROCm_GPU,measurement_type=real_rocm_gpu` 的真实 GPU 数据。

判断结果时按下面规则：

| `torch.version.hip` | `torch.version.cuda` | 含义 |
|---------------------|----------------------|------|
| 非空 | 通常为空 | ROCm 版 PyTorch，可用于 AMD GPU 实测 |
| 空 | 非空 | CUDA 版 PyTorch，装错了，不能作为 AMD ROCm 结果 |
| 空 | 空 | CPU 版或未启用 GPU，只能做 CPU baseline |

---

## 课程要求对应关系

| 课程要求 | 本项目对应实现 | 说明 |
|----------|----------------|------|
| 本地 AI 推理 | 文档问答全流程本地完成，无需联网 | 文档解析、向量检索、答案生成均在本地执行 |
| 端到端应用 | 上传文档 -> 切块 -> 嵌入 -> 检索 -> 回答 -> 资源调度展示 | 完整 RAG Pipeline + Gradio UI |
| 异构资源分工 | CPU / GPU / NPU 后端抽象与调度策略 | CPU 已真实执行；ROCm 工具已能检测；GPU 实测需安装 PyTorch-HIP；NPU 仍为接口/检测层 |
| 基础异构实验 | 矩阵乘法、FP32/FP16、MLP 训练 | 生成评分表要求的 CSV 与图表；ROCm 可用时自动加入 GPU 实测 |
| 性能与能效 | 延迟 benchmark + CPU/内存/ROCm 功耗采样 | `resource_monitor.py` 生成 `power_trace.csv` 与 `energy_summary.csv`；有 `rocm-smi` 时采集 GPU power |

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
| AMDGPUBackend | 需 ROCm PyTorch | ROCm 工具存在不等于 PyTorch-HIP 可用；需 `torch.version.hip` 非空且 `torch.cuda.is_available()` 为 True |
| AMDNPUBackend | 检测/接口层 | 当前仅检测 Ryzen AI / ONNX EP；未接入真实 ONNX NPU 推理模型 |
| SimulatedNPUBackend | 仅仿真 | 仅用于演示，不产生真实硬件结果 |

---

## 安装方法

### 环境要求

- **Python**: >= 3.9
- **操作系统**: Linux (推荐 Ubuntu 22.04+) / macOS / Windows
- **硬件** (可选): AMD Ryzen AI MAX+ 处理器（无此硬件时自动使用 CPU fallback 模式）

### 推荐安装方式

```bash
# 1. 克隆项目（或进入已有目录）
cd localdoc-agent-amd-ai

# 2. 一键运行。脚本会自动准备 Python 环境并安装基础依赖。
bash run_all_experiments.sh --quick
```

脚本的环境初始化逻辑：

1. 优先使用 `python -m venv .venv`。
2. 如果 Jupyter/Ubuntu 容器缺少 `ensurepip` 或 `python3-venv`，自动尝试 `pip install --user virtualenv`。
3. 如果 `virtualenv` 也不可用，则退回当前用户 Python 环境，用 `pip install --user` 安装依赖。

因此在无 sudo 的 Jupyter 容器里也可以直接运行：

```bash
git pull
rm -rf .venv
bash run_all_experiments.sh --quick
```

### AMD ROCm 平台依赖安装说明

基础实验和默认 Demo 不需要安装 PyTorch；它们可以直接运行。如果要启用 ROCm GPU 实测或本地 LLM，必须显式安装 ROCm 版 PyTorch。

不要使用这些命令安装 AMD 平台依赖：

```bash
pip install torch
pip install "torch>=2.2"
pip install -r requirements-llm.txt  # 旧版文件曾包含 torch；当前已修正为不含 torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

AMD ROCm 平台应使用：

```bash
# 默认使用 PyTorch ROCm 6.4 wheel 索引
bash scripts/setup_llm.sh --rocm
```

如果你的 ROCm 版本不是 6.4，需要改用匹配版本的索引，例如：

```bash
LOCALDOC_TORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3 \
  bash scripts/setup_llm.sh --rocm
```

如果已经按旧命令装成 CUDA 版 PyTorch，直接运行下面命令修正。脚本会卸载 `torch/torchvision/torchaudio`，并清理 `nvidia-*` CUDA wheel 残留依赖：

```bash
bash scripts/setup_llm.sh --rocm
```

安装后必须验证：

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

正确的 AMD ROCm 结果应满足：`torch.version.hip` 非空，且 `torch.cuda.is_available()` 为 `True`。如果 `torch.version.cuda` 非空但 `torch.version.hip` 为空，就是 CUDA 版 PyTorch，不能写成 AMD ROCm 实测。

参考官方说明时，应在 PyTorch 安装选择器中选择 `Linux + Pip + Python + ROCm`，不要选择 CUDA；AMD 官方文档也强调需选择与 Python/Ubuntu/ROCm 版本兼容的 ROCm wheel。参考链接：[PyTorch Get Started](https://pytorch.org/get-started/locally/)、[AMD ROCm PyTorch 安装文档](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.4/docs/install/installrad/native_linux/install-pytorch.html)。

---

## 快速开始

### 运行交互式 Demo

```bash
bash run_demo.sh
```

启动 Gradio Web UI，默认访问地址：`http://localhost:7860`。脚本使用 `python -m localdoc.app` 模块方式启动，并设置项目根目录到 `PYTHONPATH`，避免在 Jupyter 容器中出现 `ModuleNotFoundError: No module named 'localdoc'`。

如果是在 Jupyter/GitHub Codespace 类环境中运行，需要从平台提供的端口转发面板打开 `7860` 端口。

### 运行基准测试

推荐使用全量实验入口：

```bash
bash run_all_experiments.sh
```

该脚本会依次运行单元测试、环境检查、基础异构实验、Agent benchmark、垂直行业流程、可选本地 LLM benchmark、能耗采样和图表生成，并生成 `results/full_experiment_run.log` 与 `results/experiment_manifest.txt`。

快速验证模式：

```bash
bash run_all_experiments.sh --quick
```

单独运行 benchmark 子流程：

```bash
bash run_benchmark.sh
```

自动执行环境检查、基础异构实验、Agent 延迟测试、资源/能耗采样、垂直行业流程复现与图表生成，结果输出到 `results/` 和 `figures/` 目录。在只有 ROCm 命令行工具、但没有 PyTorch-HIP 的环境下，ROCm GPU 行会标记为 `unavailable`；SimulatedNPU 数据会标记为 `simulated`。

常用参数：

```bash
# 快速 smoke test，适合改代码后验证
bash run_benchmark.sh --quick

# 只跑评分表基础实验：matmul / FP16 / MLP
bash run_benchmark.sh --basic-only

# 额外运行本地 LLM 生成 benchmark（默认只使用本地模型，不联网下载）
bash run_benchmark.sh --with-llm

# 允许 benchmark 阶段从 Hugging Face Hub 拉取模型
bash run_benchmark.sh --allow-llm-hub
```

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
| `results/rocminfo.txt` / `results/rocm_smi.txt` / `results/hipcc_version.txt` / `results/hipconfig_full.txt` | ROCm 命令行原始证据；无 ROCm 时记录 `COMMAND NOT FOUND` |
| `results/matmul_benchmark.csv` | CPU/ROCm GPU 矩阵乘法 benchmark（平均时间、标准差、加速比） |
| `results/precision_compare.csv` | FP32/FP16 耗时、加速比、最大/平均绝对误差 |
| `results/mlp_train_log.csv` | MLP 前向、反向、参数更新训练日志（loss、accuracy、epoch time） |
| `results/latency_results.csv` | 延迟基准测试结果（CSV 中 `measurement_type` 列区分 real/simulated） |
| `results/backend_results.csv` | 多后端对比结果（含 `real_inference` 字段） |
| `results/resource_usage.csv` | 系统资源使用快照 |
| `results/power_trace.csv` / `results/energy_summary.csv` | CPU/内存/ROCm GPU 功耗采样与能耗估算 |
| `results/vertical_demo_transcript.csv` | 企业内网政策问答端到端演示 transcript，含问题、回答、来源、分数、调度 trace |
| `results/llm_generation_benchmark.csv` | 可选本地 LLM 生成 benchmark；模型缺失时写入 skipped 记录 |
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
├── run_all_experiments.sh              # 一键全量实验：测试+benchmark+图表+manifest
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
│   ├── check_environment.py            # 环境检测（硬件/驱动/后端/内核）
│   ├── basic_benchmarks.py             # 基础实验：matmul / FP16 / MLP
│   ├── resource_monitor.py             # CPU/内存/ROCm GPU 功耗采样
│   ├── demo_vertical_workflow.py        # 垂直行业端到端 QA transcript
│   ├── plot_basic_results.py           # 基础实验绘图
│   ├── benchmark_real.py               # 真实硬件基准测试（自动检测 GPU/NPU）
│   ├── benchmark_latency.py            # 模拟延迟基准测试（simulated only）
│   ├── benchmark_llm_generation.py     # LLM 生成延迟测试
│   ├── benchmark_rag_modes.py          # RAG 模式对比（extractive vs LLM）
│   ├── plot_results.py                 # 基础结果绘图
│   └── plot_llm_results.py             # LLM 结果绘图
│
├── tests/                              # 单元测试
│   ├── __init__.py
│   ├── conftest.py                     # 共享 fixtures
│   ├── test_chunker.py                 # 文本切块测试
│   ├── test_retriever.py               # 语义检索测试
│   ├── test_scheduler.py               # 异构调度器测试
│   └── test_backend_consistency.py     # 后端向量维度与生成 fallback 一致性测试
│
├── scripts/                            # 辅助脚本
│   ├── bootstrap_python_env.sh          # 无 sudo Jupyter 环境 Python 初始化
│   ├── download_llm.py                  # 下载/选择本地 LLM
│   ├── download_llm.sh                  # LLM 下载入口
│   ├── setup_llm.sh                     # LLM 依赖安装；显式选择 --rocm/--cpu
│   ├── run_demo_llm.sh                  # 启用本地 LLM 的 Demo
│   └── run_llm_benchmark.sh             # 本地 LLM benchmark
│
├── results/                            # 实验结果 CSV
│   ├── environment_report.txt          # 环境检测报告
│   ├── matmul_benchmark.csv            # 矩阵乘法基础实验
│   ├── precision_compare.csv           # FP32/FP16 精度对比
│   ├── mlp_train_log.csv               # MLP 训练日志
│   ├── latency_results.csv             # 延迟测试结果
│   ├── backend_results.csv             # 后端对比结果
│   ├── resource_usage.csv              # 资源使用快照
│   ├── power_trace.csv                 # 资源/功耗采样时间序列
│   ├── energy_summary.csv              # 能耗估算摘要
│   └── vertical_demo_transcript.csv    # 企业内网应用流程复现记录
│
├── figures/                            # 实验图表
│   ├── latency_comparison.png          # 延迟对比图
│   ├── matmul_benchmark.png            # 矩阵乘法耗时图
│   ├── precision_compare.png           # FP32/FP16 对比图
│   ├── mlp_training_curve.png          # MLP loss/accuracy 曲线
│   ├── backend_comparison.png          # 后端性能对比图
│   ├── energy_comparison.png           # 资源/功耗采样图
│   └── resource_usage.png              # 资源使用图
│
├── examples/                           # 垂直行业演示材料
│   └── enterprise_policy/              # 企业内网政策/应急处置示例文档
│
└── docs/                               # 文档目录
    ├── system_design.md                # 系统设计文档
    ├── reproduction.md                 # 实验复现指南
    ├── amd_ai_max_backend.md           # AMD 硬件后端替换指南
    ├── screenshot_checklist.md         # 报告/答辩截图清单
    └── experiment_report_draft.md      # 实验报告草稿 (中文)
```

---

## 当前限制说明

1. **AMD 平台已检测到 ROCm 工具，但 PyTorch-HIP 需要单独安装**：最新 AMD/Jupyter 运行记录中 `rocminfo`、`rocm-smi`、`hipcc`、`hipconfig` 可用，且检测到 `gfx1151`；初始记录里 `torch 已安装: False`。如果后续按旧依赖装成 CUDA 版 PyTorch，也仍然不能激活 ROCm GPU benchmark，必须重装 ROCm 版 PyTorch。

2. **NPU 后端仍是检测/接口层**：当前 `AMDNPUBackend` 能检测 ONNX Runtime EP，但没有真实 ONNX NPU 推理模型；即使检测到 EP，也会在 benchmark 中标记为 `cpu_fallback_with_hardware_detected`，不会标为 `real_hardware`。

3. **TF-IDF 嵌入（非神经网络）**：当前向量嵌入模块使用 TF-IDF，而非神经网络嵌入模型。在真实 AMD GPU/NPU 环境下可替换为 Dense Embedding 模型以利用硬件加速。

4. **模板式答案生成**：当前答案生成模块采用抽取式/模板式策略，未接入大语言模型。在真实硬件平台上可接入 Qwen3.5 4B 等模型实现神经网络生成。

5. **文档格式有限**：当前支持 PDF、TXT、Markdown 三种格式，其他格式（如 DOCX、HTML）需扩展 Loader 模块。

---

## 后续：启用 ROCm GPU 实测

AMD 平台已经能运行 ROCm 工具链。下一步不是“找硬件”，而是在 Python 环境中安装匹配 ROCm 版本的 PyTorch-HIP。

### 第一步：保留当前 ROCm 环境证据

```bash
bash run_all_experiments.sh --quick
```

截图和保存以下文件：

- `results/environment_report.txt`
- `results/rocminfo.txt`
- `results/rocm_smi.txt`
- `results/hipcc_version.txt`
- `results/hipconfig_full.txt`

### 第二步：安装 ROCm 版 PyTorch

先不要直接 `pip install torch`。本项目的安全入口是：

```bash
bash scripts/setup_llm.sh --rocm
```

该脚本会使用 ROCm PyTorch wheel 索引，并在安装前卸载可能误装的 CUDA 版 torch。默认索引为 `https://download.pytorch.org/whl/rocm6.4`；如 AMD/Jupyter 环境的 ROCm 版本不同，用 `LOCALDOC_TORCH_ROCM_INDEX_URL` 指定匹配版本。

也可以根据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 或 [AMD ROCm PyTorch 安装文档](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.4/docs/install/installrad/native_linux/install-pytorch.html) 选择与当前 ROCm、Python、Ubuntu 版本匹配的安装命令。安装后必须验证：

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### 第三步：重新跑完整实验

```bash
bash run_all_experiments.sh --quick
# 或正式数据：
bash run_all_experiments.sh
```

预期变化：

- `results/environment_report.txt` 中 `torch.version.hip` 非空。
- `torch.cuda.is_available()` 为 `True`。
- `torch.version.cuda` 不应作为 AMD 证据；如果 `torch.version.cuda` 非空但 `torch.version.hip` 为空，说明装成 CUDA 版。
- `results/matmul_benchmark.csv`、`results/precision_compare.csv`、`results/mlp_train_log.csv` 出现 `ROCm_GPU` 且 `measurement_type=real_rocm_gpu`。
- `figures/matmul_benchmark.png`、`figures/precision_compare.png`、`figures/mlp_training_curve.png` 中出现 ROCm GPU 曲线或柱状对比。

### 第四步：更新实验报告

只有当 CSV 中出现 `real_rocm_gpu` 或 `real_hardware` 后，报告中才能写“ROCm GPU 实测”。如果仍是 `unavailable` / `simulated`，报告只能写“AMD 平台环境检测 + CPU baseline + simulated backend”。

### 第五步：接入 Qwen3-1.7B 本地 LLM（可选）

在真实 AMD 平台上，有多种方式运行本地 LLM：

1. **Lemonade 推理框架**（推荐）：AMD 官方推理框架，内置 NPU 调度，提供 OpenAI 兼容 API。
   - 安装：`pip install lemonade-sdk`
   - AMD 官方适配模型：https://huggingface.co/amd

2. **LM Studio + ROCm**：使用 LM Studio 加载量化模型，通过 ROCm GPU 加速推理。
   - 下载：https://lmstudio.ai/

3. **Ryzers Docker 容器**：AMD Research 提供的预配置容器，覆盖 LLM/NPU/视觉等场景。
   - 仓库：https://github.com/AMDResearch/Ryzers

4. **直接使用 Transformers + ROCm PyTorch**：当前 `local_llm_backend.py` 已支持自动检测 ROCm GPU。

---

## 可选：接入本地小语言模型

默认版本不加载 LLM，保证普通 CPU 环境可运行。如果需要展示**本地 AI 推理**能力，可以启用 Qwen3-1.7B 作为生成后端。

### 启用步骤

```bash
# 1. 安装 LLM 依赖
# AMD ROCm 平台：
bash scripts/setup_llm.sh --rocm

# 普通 CPU 环境：
# bash scripts/setup_llm.sh --cpu

# 2. 下载模型（约 1GB，从 Hugging Face）
bash scripts/download_llm.sh

# 3. 测试模型
python scripts/test_llm.py

# 4. 启动带 LLM 的 Demo
bash scripts/run_demo_llm.sh

# 5. 运行 LLM Benchmark（可选）
bash scripts/run_llm_benchmark.sh
```

### 说明

- 本项目使用 **Qwen3-1.7B** 作为可选本地 LLM 后端。
- `requirements-llm.txt` 不再包含 `torch`，避免在 AMD 平台误装 CUDA 版 PyTorch。
- AMD 平台必须用 `bash scripts/setup_llm.sh --rocm` 安装 ROCm 版 PyTorch；普通 `pip install torch` 不可作为 AMD 实测依赖安装方式。
- 默认不加载 LLM，设置 `LOCALDOC_USE_LLM=1` 才启用。
- Qwen3-1.7B 用于展示本地生成式 AI 推理能力。
- 关闭 thinking mode（`enable_thinking=False`）以保证演示稳定快速。
- 该 LLM **完全本地运行**，不调用任何云端 API。
- Embedding 仍使用 TF-IDF（CPUBackend），LLM 仅替换答案生成环节。
- 如果没有真实 AMD ROCm/NPU 环境，则该实验**不是** AMD GPU/NPU 加速实验。
- 模型可从 `Qwen/Qwen3-1.7B`（Hugging Face）下载，Apache-2.0 许可证。
- 详细说明见 [docs/local_llm_setup.md](docs/local_llm_setup.md)。

---

## 许可证

本项目采用 [MIT License](LICENSE)，仅供学术研究与课程实验使用。
