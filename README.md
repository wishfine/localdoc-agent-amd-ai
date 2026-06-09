# 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体设计与异构资源调度仿真实验

![课程实验项目](https://img.shields.io/badge/异构计算-课程实验项目-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 项目定位

这是一个面向 AMD Ryzen AI MAX+ 平台设计的本地知识库智能体原型与异构资源调度实验项目。项目实现了完整的 RAG（Retrieval-Augmented Generation）管线：文档加载、智能切块、向量嵌入、语义检索、答案生成，并通过统一后端接口和调度器组织 CPU / GPU / NPU 三类异构资源。

当前代码已经支持在普通 CPU 环境、无 sudo 的 Jupyter 容器环境、以及 AMD ROCm Linux 环境中运行。实验脚本会自动区分真实 CPU、真实 ROCm GPU、CPU fallback、SimulatedNPU 和 unavailable 后端，并在 CSV 中用 `measurement_type`、`is_simulated`、`real_inference` 明确标注，避免把模拟数据写成真实硬件数据。

---

## 运行方式（先看这里）

本项目默认要接入本地 LLM。AMD 平台上按下面顺序执行即可：先安装依赖和 ROCm 版 PyTorch，再下载 Qwen3-1.7B 模型，最后一键跑完所有实验。

### 1. 安装所有依赖并下载本地 LLM

AMD ROCm 平台使用这一组命令：

```bash
cd ~/localdoc-agent-amd-ai
git pull
rm -rf .venv

# 安装基础依赖、LLM 依赖、ROCm 版 PyTorch；不要直接 pip install torch 或 pip install -r requirements-llm.txt
bash scripts/setup_llm.sh --rocm

# 下载本地 Qwen3-1.7B 模型到 models/qwen3-1.7b/
bash scripts/download_llm.sh

# 验证本地 LLM 能加载并生成
python scripts/test_llm.py
```

如果 Hugging Face 下载慢，可先设置镜像后再下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_llm.sh
```

普通 CPU 环境仅用于开发验证时，可把第一条安装命令改成：

```bash
bash scripts/setup_llm.sh --cpu
```

### 2. 一键跑完所有实验结果

安装依赖和下载模型后，运行：

```bash
bash run_all_experiments.sh --allow-llm-hub
```

该命令会一次性生成所有实验结果：

- 单元测试
- 环境检查和 ROCm 原始证据
- ROCm 官方工具性能检测与调优证据：AMD SMI、ROCm SMI、Bandwidth Test、rocprofv3/ROCProfiler
- 矩阵乘法 benchmark
- FP32/FP16 精度与性能对比
- MLP 前向/反向/参数更新训练实验
- Agent embedding / query / generation / end-to-end RAG 延迟测试
- 企业内网政策问答端到端 transcript
- CPU/内存/ROCm GPU 资源与能耗采样
- 本地 Qwen3-1.7B LLM 生成 benchmark
- extractive RAG 与 local LLM RAG 模式对比
- ROCm 工具汇总表 `results/rocm_tools_summary.csv` 与调优记录 `results/rocm_tuning_recommendations.md`
- 所有 CSV、PNG 图表、`results/full_experiment_run.log`、`results/experiment_manifest.txt`

### 3. 启动本地 LLM Demo

实验跑完后，如需截图 Web 演示页：

```bash
bash scripts/run_demo_llm.sh
```

打开平台转发的 `7860` 端口，截图上传文档、构建知识库、提问、LLM 回答和调度日志。

### 4. 验证 ROCm 是否真实跑通

跑完后检查：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

grep "ROCm_GPU" results/matmul_benchmark.csv
grep "ROCm_GPU" results/precision_compare.csv
grep "ROCm_GPU" results/mlp_train_log.csv
head results/llm_generation_benchmark.csv
head results/rocm_tools_summary.csv
sed -n '1,160p' results/rocm_tuning_recommendations.md
```

判定规则：

| `torch.version.hip` | `torch.version.cuda` | 能否写 AMD ROCm 实测 |
|---------------------|----------------------|----------------------|
| 非空 | 通常为空 | 可以，CSV 应出现 `measurement_type=real_rocm_gpu` |
| 空 | 非空 | 不可以，这是 CUDA 版 PyTorch，装错了 |
| 空 | 空 | 不可以，只能作为 CPU baseline |

### 5. 推送实验结果

确认结果正确后推送到远程仓库。不要提交 `models/`，模型文件很大，已经被 `.gitignore` 忽略。

```bash
git pull --rebase origin main
git status --short

git add \
  results/environment_report.txt \
  results/rocminfo.txt results/rocm_smi.txt results/hipcc_version.txt results/hipconfig_full.txt \
  results/rocm_tools_summary.csv results/rocm_tuning_recommendations.md \
  results/amd_smi_list.txt results/amd_smi_static.txt results/amd_smi_metric.txt \
  results/rocm_smi_performance.txt results/rocm_bandwidth_test.txt \
  results/rocprofiler_tools.txt results/rocprofiler_run.txt \
  results/matmul_benchmark.csv results/precision_compare.csv results/mlp_train_log.csv \
  results/latency_results.csv results/backend_results.csv results/resource_usage.csv \
  results/power_trace.csv results/energy_summary.csv \
  results/vertical_demo_transcript.csv results/llm_generation_benchmark.csv \
  results/rag_mode_comparison.csv results/rag_stage_breakdown.csv \
  results/full_experiment_run.log results/experiment_manifest.txt \
  figures/matmul_benchmark.png figures/precision_compare.png figures/mlp_training_curve.png \
  figures/energy_comparison.png figures/latency_comparison.png figures/backend_comparison.png \
  figures/resource_usage.png figures/llm_generation_latency.png \
  figures/rag_mode_comparison.png figures/rag_stage_breakdown.png

git commit -m "Add AMD full experiment results"
git push origin main
```

**实验诚信说明**：本项目不伪造 AMD GPU/NPU 结果。只有 CSV 中出现 `real_rocm_gpu` 或 `real_hardware` 时，报告中才写真实硬件实测；`simulated`、`unavailable`、`cpu_fallback_with_hardware_detected` 不能写成真实 AMD 加速结果。

---

## 课程要求对应关系

| 课程要求 | 本项目对应实现 | 说明 |
|----------|----------------|------|
| 本地 AI 推理 | 文档问答全流程本地完成，无需联网 | 文档解析、向量检索、本地 Qwen3-1.7B 生成均在本机执行 |
| 端到端应用 | 上传文档 -> 切块 -> 嵌入 -> 检索 -> 本地 LLM 回答 -> 资源调度展示 | 完整 RAG Pipeline + Gradio UI |
| 异构资源分工 | CPU / GPU / NPU 后端抽象与调度策略 | CPU 已真实执行；ROCm 工具已能检测；GPU 实测需安装 PyTorch-HIP；NPU 仍为接口/检测层 |
| 基础异构实验 | 矩阵乘法、FP32/FP16、MLP 训练 | 生成评分表要求的 CSV 与图表；ROCm 可用时自动加入 GPU 实测 |
| 性能与能效 | 延迟 benchmark + CPU/内存/ROCm 功耗采样 + ROCm 官方工具证据 | `resource_monitor.py` 生成 `power_trace.csv` 与 `energy_summary.csv`；`rocm_tools_profile.py` 采集 AMD SMI、ROCm SMI、Bandwidth Test、rocprofv3 证据并生成调优记录 |

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

## 命令索引（排错用）

正常情况下只需要执行顶部的三条核心命令：`setup_llm.sh --rocm`、`download_llm.sh`、`run_all_experiments.sh --allow-llm-hub`。下面命令只用于单项排错。

| 目标 | 命令 |
|------|------|
| 只做快速 smoke test | `bash run_all_experiments.sh --quick` |
| 只跑基础实验 | `bash run_benchmark.sh --basic-only` |
| 只跑 Agent benchmark | `bash run_benchmark.sh --agent-only` |
| 只跑本地 LLM/RAG benchmark | `bash scripts/run_llm_benchmark.sh` |
| 启动带本地 LLM 的 Web Demo | `bash scripts/run_demo_llm.sh` |
| 启动抽取式 Web Demo | `bash run_demo.sh` |
| 单独检查环境 | `python experiments/check_environment.py` |
| 单独采集 ROCm 工具证据 | `python experiments/rocm_tools_profile.py` |
| 单独跑 matmul / FP16 / MLP | `python experiments/basic_benchmarks.py` |
| 单独跑 Agent 真实/模拟 benchmark | `python experiments/benchmark_real.py` |
| 单独跑企业内网 QA transcript | `python experiments/demo_vertical_workflow.py` |
| 单独生成图表 | `python experiments/plot_basic_results.py && python experiments/plot_results.py && python experiments/plot_llm_results.py` |
| 单元测试 | `python -m pytest tests/ -v` |

---

## 输出文件说明

| 文件路径 | 说明 |
|----------|------|
| `results/environment_report.txt` | 环境检测报告（硬件、驱动、后端可用性） |
| `results/rocminfo.txt` / `results/rocm_smi.txt` / `results/hipcc_version.txt` / `results/hipconfig_full.txt` | ROCm 命令行原始证据；无 ROCm 时记录 `COMMAND NOT FOUND` |
| `results/rocm_tools_summary.csv` | ROCm 官方工具采集汇总：AMD SMI、ROCm SMI、Bandwidth Test、Profiler 的可用性、命令、输出文件和用途 |
| `results/rocm_tuning_recommendations.md` | 基于 ROCm 工具和实验结果的调优说明，可直接用于报告“优化分析”部分 |
| `results/amd_smi_list.txt` / `results/amd_smi_static.txt` / `results/amd_smi_metric.txt` | AMD SMI 原始输出：GPU 列表、静态信息、功耗/温度/显存/利用率等指标 |
| `results/rocm_smi_performance.txt` | ROCm SMI 原始输出：兼容旧环境的 GPU power、clock、VRAM、temperature、utilization 证据 |
| `results/rocm_bandwidth_test.txt` | ROCm Bandwidth Test 原始输出，用于说明 CPU-GPU/GPU-GPU 数据传输带宽 |
| `results/rocprofiler_tools.txt` / `results/rocprofiler_run.txt` | ROCProfiler/rocprofv3 工具可用性与小型 ROCm PyTorch profiler probe 输出 |
| `results/matmul_benchmark.csv` | CPU/ROCm GPU 矩阵乘法 benchmark（平均时间、标准差、加速比） |
| `results/precision_compare.csv` | FP32/FP16 耗时、加速比、最大/平均绝对误差 |
| `results/mlp_train_log.csv` | MLP 前向、反向、参数更新训练日志（loss、accuracy、epoch time） |
| `results/latency_results.csv` | 延迟基准测试结果（CSV 中 `measurement_type` 列区分 real/simulated） |
| `results/backend_results.csv` | 多后端对比结果（含 `real_inference` 字段） |
| `results/resource_usage.csv` | 系统资源使用快照 |
| `results/power_trace.csv` / `results/energy_summary.csv` | CPU/内存/ROCm GPU 功耗采样与能耗估算 |
| `results/vertical_demo_transcript.csv` | 企业内网政策问答端到端演示 transcript，含问题、回答、来源、分数、调度 trace |
| `results/llm_generation_benchmark.csv` | 本地 Qwen3-1.7B 生成 benchmark |
| `results/rag_mode_comparison.csv` / `results/rag_stage_breakdown.csv` | extractive RAG 与 local LLM RAG 对比 |
| `figures/*.png` | 性能、能耗、LLM 与 RAG 对比图表 |

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
│   ├── rocm_tools_profile.py           # ROCm 工具性能检测与调优证据
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
│   ├── rocm_tools_summary.csv          # ROCm 工具采集汇总
│   ├── rocm_tuning_recommendations.md  # ROCm 工具调优记录
│   ├── amd_smi_list.txt                # AMD SMI GPU 列表
│   ├── amd_smi_static.txt              # AMD SMI 静态硬件信息
│   ├── amd_smi_metric.txt              # AMD SMI 运行指标
│   ├── rocm_smi_performance.txt        # ROCm SMI 性能/功耗指标
│   ├── rocm_bandwidth_test.txt         # ROCm Bandwidth Test 输出
│   ├── rocprofiler_tools.txt           # ROCProfiler 工具可用性
│   ├── rocprofiler_run.txt             # rocprofv3 小型 profile 输出
│   ├── matmul_benchmark.csv            # 矩阵乘法基础实验
│   ├── precision_compare.csv           # FP32/FP16 精度对比
│   ├── mlp_train_log.csv               # MLP 训练日志
│   ├── latency_results.csv             # 延迟测试结果
│   ├── backend_results.csv             # 后端对比结果
│   ├── resource_usage.csv              # 资源使用快照
│   ├── power_trace.csv                 # 资源/功耗采样时间序列
│   ├── energy_summary.csv              # 能耗估算摘要
│   ├── vertical_demo_transcript.csv    # 企业内网应用流程复现记录
│   ├── llm_generation_benchmark.csv    # 本地 LLM 生成延迟
│   ├── rag_mode_comparison.csv         # RAG 模式对比
│   └── rag_stage_breakdown.csv         # RAG 阶段耗时分解
│
├── figures/                            # 实验图表
│   ├── latency_comparison.png          # 延迟对比图
│   ├── matmul_benchmark.png            # 矩阵乘法耗时图
│   ├── precision_compare.png           # FP32/FP16 对比图
│   ├── mlp_training_curve.png          # MLP loss/accuracy 曲线
│   ├── backend_comparison.png          # 后端性能对比图
│   ├── energy_comparison.png           # 资源/功耗采样图
│   ├── resource_usage.png              # 资源使用图
│   ├── llm_generation_latency.png      # 本地 LLM 生成延迟图
│   ├── rag_mode_comparison.png         # RAG 模式对比图
│   └── rag_stage_breakdown.png         # RAG 阶段耗时分解图
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

1. **ROCm PyTorch 必须装对**：AMD 平台不要直接 `pip install torch`，否则可能装成 CUDA 版。必须用 `bash scripts/setup_llm.sh --rocm` 或按 PyTorch 官方选择 `Linux + Pip + Python + ROCm` 的 wheel。

   也不要直接执行 `pip install -r requirements-llm.txt`：`accelerate` 会传递依赖 `torch`，pip 可能先从 PyPI 拉到 CUDA 版 torch。`scripts/setup_llm.sh --rocm` 已改成先装 ROCm torch，再装 `accelerate`。

2. **NPU 仍是检测/接口层**：当前 `AMDNPUBackend` 能检测 ONNX Runtime EP，但没有真实 ONNX NPU 推理模型；即使检测到 EP，也会在 benchmark 中标记为 `cpu_fallback_with_hardware_detected`，不会标为 `real_hardware`。

3. **Embedding 仍使用 TF-IDF**：文档嵌入和检索保持轻量实现；本地 LLM 主要用于答案生成与 RAG 模式对比。

4. **LLM 模型不进仓库**：`models/` 被 `.gitignore` 忽略。实验结果 CSV、PNG、log 和 manifest 可以提交，模型权重不要提交。

5. **真实/模拟必须区分**：只有 CSV 中出现 `real_rocm_gpu` 或 `real_hardware` 后，报告中才写真实硬件实测；`simulated`、`unavailable`、`cpu_fallback_with_hardware_detected` 不能写成真实 AMD 加速。

---

## 许可证

本项目采用 [MIT License](LICENSE)，仅供学术研究与课程实验使用。
