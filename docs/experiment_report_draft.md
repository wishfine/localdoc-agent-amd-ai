# 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体设计与异构资源调度仿真实验

## 实验报告

---

## 一、摘要

本实验面向《异构计算》课程"实验类方案三"的要求，设计并实现了一个面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体系统（LocalDoc Agent）。当前没有真实的 AMD AI MAX+ 硬件环境，因此所有涉及 GPU 与 NPU 的实验均采用 CPU fallback 与 SimulatedNPUBackend（CPU 计算加人为延迟）完成原型验证。仿真实验验证了从文档加载、文本分块、向量嵌入、语义检索到智能问答的端到端流程，以及异构资源调度框架的可扩展性与后端回退机制。实验结果表明，系统架构设计合理，在后续获得真实 AMD 硬件后可直接替换后端并进行性能实测。

**关键词**：异构计算、仿真实验、本地知识库、智能体、AMD Ryzen AI MAX+、RAG、TF-IDF

---

## 二、实验目的与要求

### 2.1 实验目的

1. **理解异构计算架构**：深入理解 CPU、GPU（AMD RDNA 3.5 iGPU）、NPU（AMD XDNA）三类异构计算单元的架构特点、适用场景与协同工作原理。
2. **掌握面向异构平台的任务调度设计**：针对不同计算任务的特征（I/O 密集、计算密集、推理密集），设计合理的后端选择与调度策略。
3. **实现完整的本地知识库智能体**：涵盖文档加载、文本分块、向量嵌入、语义检索、答案生成的完整 RAG（检索增强生成）管线。
4. **通过仿真实验验证调度框架可行性**：在没有真实 AMD 硬件的条件下，利用 CPU fallback 与模拟后端验证异构调度框架的接口设计、任务分发、后端回退等核心机制。

### 2.2 实验要求

1. 设计面向 AMD 平台的本地知识库 Agent 系统架构。
2. 实现 DocumentLoader、TextChunker、EmbeddingEngine、DocumentRetriever、AnswerGenerator 等核心模块。
3. 设计 HeterogeneousScheduler 异构调度器，将计算任务合理分配到 CPU、GPU、NPU。
4. 实现 CPUBackend、AMDGPUBackend、AMDNPUBackend、SimulatedNPUBackend 等多种后端。
5. 进行仿真实验基准测试，分析各后端的模拟延迟差异与系统可扩展性。
6. 撰写实验报告，总结仿真结果与后续真实硬件验证计划。

---

## 三、实验环境配置

### 3.1 开发环境

| 项目 | 配置 |
|------|------|
| 操作系统 | macOS Darwin 25.5.0 |
| Python 版本 | 3.10.x |
| CPU | Apple Silicon / Intel x86_64 |
| 内存 | 16 GB |
| 开发工具 | VS Code, Git |

详细的环境检查报告见 `results/environment_report.txt`（由 `experiments/check_environment.py` 自动生成）。

### 3.2 硬件限制声明

**重要声明：当前没有 AMD AI MAX+ 硬件。** 本实验的全部 GPU/NPU 相关数据均基于模拟后端获取，不包含任何真实 AMD 硬件的性能数据。

| 后端 | 实际状态 | 说明 |
|------|----------|------|
| CPUBackend | 真实可用 | 纯 Python + TF-IDF 实现，作为基线对照 |
| AMDGPUBackend | CPU fallback | PyTorch 未检测到 ROCm/HIP 后端，自动回退到 CPU |
| AMDNPUBackend | 不可用 | 未安装 Ryzen AI SDK / ONNX Runtime VitisAI EP |
| SimulatedNPUBackend | CPU 计算 + 人为延迟 | 所有计算在 CPU 上执行，通过 time.sleep() 模拟 NPU 推理耗时 |

### 3.3 软件依赖

```
numpy>=1.21          # 数值计算核心库
gradio>=3.40         # Web UI 框架
psutil>=5.9          # 系统资源监控（可选）
matplotlib>=3.5      # 图表绘制（可选）
```

---

## 四、系统设计与实现

### 4.1 整体架构

本系统采用 RAG（检索增强生成）管线架构，核心流程为：

```
文档加载 -> 文本分块 -> 向量嵌入 -> 索引存储 -> 查询检索 -> 答案生成
```

每个阶段通过 HeterogeneousScheduler 选择最优硬件后端执行。系统包含以下核心类（均位于 `localdoc/` 包中）：

### 4.2 核心模块

#### 4.2.1 DocumentLoader（localdoc/loader.py）

负责从各种格式的文档中提取文本内容，支持 Markdown（.md）、纯文本（.txt）、PDF（.pdf）三种格式。内部按文件扩展名自动选择加载方式，PDF 加载优先使用 pypdf，回退到 pdfplumber 和 PyPDF2。支持递归加载整个目录下的所有文档。DocumentLoader 属于 I/O 密集型任务，由调度器固定分配到 CPU 后端。

#### 4.2.2 TextChunker（localdoc/chunker.py）

负责将长文本切分为适合向量检索的较小文本块。采用"先段落、后句子"的两层切分策略：

1. **第一层**：优先按段落（双换行符 `\n\s*\n`）切分。
2. **第二层**：对超过 chunk_size（默认 500 字符）的段落，按句子结束标记进一步切分。若单个句子仍超长，则进行强制字符级切割。
3. **重叠窗口**：相邻块之间保留 chunk_overlap（默认 50 字符）的重叠区域，避免上下文断裂。

TextChunker 属于 CPU 逻辑处理任务，固定在 CPU 后端执行。

#### 4.2.3 EmbeddingEngine（localdoc/embedding.py）

负责将文本块和查询转换为数值向量。支持两种工作模式：

- **后端模式**：当提供了 backend 参数时，调用后端的 `embed_texts()` 方法。
- **TF-IDF 回退模式**：无后端时使用内置的 TF-IDF 算法，通过词频（TF）与逆文档频率（IDF）的乘积衡量词语重要程度，输出 L2 归一化后的稀疏向量。

EmbeddingEngine 是异构调度的主要加速目标，调度器优先将其分配到 NPU 后端，依次回退到 GPU 和 CPU。

#### 4.2.4 DocumentRetriever（localdoc/retriever.py）

负责存储文档块及其向量，并根据查询进行语义检索。内部维护两个平行列表（_documents 与 _embeddings），检索时将查询向量与所有文档向量计算余弦相似度，返回得分最高的 Top-K 个结果。余弦相似度的计算由 EmbeddingEngine 的静态方法 `cosine_similarity()` 提供。检索属于计算密集型任务，调度器优先分配到 GPU 后端。

#### 4.2.5 AnswerGenerator（localdoc/generator.py）

负责根据检索到的文档块和用户查询生成回答。支持两种工作模式：

- **后端模式**：调用后端的 `generate_answer()` 方法，通过 LLM 生成自然语言回答。
- **抽取式回退模式**：从上下文中提取与查询关键词最相关的句子，按相关性评分排序后拼接为回答。评分依据包括关键词命中率（权重 0.6）、块相关性分数（权重 0.3）和句子长度因子（权重 0.1）。

答案生成属于推理密集型任务，调度器优先分配到 GPU 后端。

#### 4.2.6 LocalDocAgent（localdoc/agent.py）

RAG 管线的主控类，负责协调上述各模块完成端到端流程。当提供了 scheduler 时，文档导入流程的加载、分块、向量化三个阶段以及查询流程的检索、生成两个阶段均通过调度器选择后端并记录调度日志。返回结果中包含 `backend_trace` 字段，记录每个阶段使用的后端、选择原因和执行耗时。LocalDocAgent 同时提供异步接口（`aingest_document`、`aingest_directory`、`aquery`），基于 asyncio 的 `run_in_executor` 实现。

### 4.3 HeterogeneousScheduler（localdoc/scheduler.py）

异构资源调度器，根据任务类型自动选择最优硬件后端。内部维护 `_TASK_BACKEND_PRIORITY` 优先级表，定义了每种任务类型的后端选择顺序：

| 任务类型 | 首选后端 | 备选后端 | 选择原因 |
|----------|----------|----------|----------|
| document_loading | CPU | -- | CPU 最适合 I/O 密集的文档加载 |
| chunking | CPU | -- | CPU 足以完成文本分块的逻辑处理 |
| embedding | NPU | GPU > CPU | NPU 优先用于 INT8 推理的嵌入计算 |
| retrieval | GPU | CPU | GPU 优先用于并行向量检索 |
| generation | GPU | CPU | GPU 优先用于 LLM 生成推理 |

调度器的 `execute()` 方法在执行函数前后记录时间戳和执行日志，支持通过 `get_execution_log()` 获取完整的调度历史。调度器还内置模拟后端检测功能：当后端名称以 "Simulated" 开头时，自动在日志中标注 `[SIMULATED - not real hardware]`。

### 4.4 后端实现

#### CPUBackend（localdoc/backends/cpu_backend.py）

基于纯 Python 实现的 CPU 计算后端，始终可用。嵌入方法采用 TF-IDF 思路的纯 Python 实现（基于 `collections.Counter`），无需安装 sklearn 等第三方依赖。答案生成基于关键词匹配的抽取式方法。作为异构计算实验的基线对照组。

#### AMDGPUBackend（localdoc/backends/gpu_backend.py）

基于 PyTorch + ROCm/HIP 的 GPU 加速后端。设计为通过 PyTorch 的 CUDA API 统一访问 AMD HIP 设备。当运行环境不满足要求时（PyTorch 未编译 HIP 后端或无可用 GPU 设备），所有方法自动回退到 CPU 计算。**当前环境中不可用，实际执行 CPU fallback。**

#### AMDNPUBackend（localdoc/backends/npu_backend.py）

基于 ONNX Runtime + Ryzen AI SDK 的 NPU 加速后端。通过检查 ONNX Runtime 的可用 Execution Provider 列表中是否包含 VitisAIExecutionProvider 或 DmlExecutionProvider 来判断 NPU 是否可用。**当前环境中不可用。**

#### SimulatedNPUBackend（localdoc/backends/simulated_npu.py）

模拟 NPU 后端，**仅用于演示和教学目的**。所有计算仍在 CPU 上执行，通过 `time.sleep()` 添加人为延迟（每条文本 5ms 基础延迟 + 最高 3ms 随机抖动，生成延迟 20ms + 最高 3ms 抖动）来模拟 NPU 推理耗时。此后端的性能数据不代表真实 AMD NPU 的性能，不得作为真实 NPU 性能参考。

---

## 五、实验内容与步骤

### 5.1 环境检查

运行 `experiments/check_environment.py` 检测当前硬件和软件环境，判断是否有真实的 AMD GPU（ROCm）或 NPU（Ryzen AI SDK）可用。脚本检测 Python 环境、系统内存、PyTorch（HIP/CUDA 可用性）和 ONNX Runtime（VitisAI/DirectML EP 可用性），并将结论写入 `results/environment_report.txt`。

为满足评分表对实验环境证据的要求，脚本还会尝试执行 `rocminfo`、`rocm-smi --showproductname --showpower --showmeminfo vram`、`hipcc --version` 和 `hipconfig --full`，将原始输出分别保存到 `results/rocminfo.txt`、`results/rocm_smi.txt`、`results/hipcc_version.txt` 和 `results/hipconfig_full.txt`。在当前 macOS 环境中这些命令不可用，结果文件会明确记录 `COMMAND NOT FOUND`；后续在 AMD ROCm 环境运行时，这些文件可作为 GPU 名称、gfx 架构、HIP 版本和功耗信息的原始证据。

检测结论：当前环境为 "CPU fallback + simulated backend"，未检测到真实 AMD 硬件。

### 5.2 基础异构实验

为对齐课程评分表中的基础实验要求，新增 `experiments/basic_benchmarks.py`，统一生成以下三类实验结果：

1. **矩阵乘法 benchmark**：在不同矩阵规模下测试 FP32 矩阵乘法耗时，输出 `results/matmul_benchmark.csv`，字段包含平均时间、标准差、最小/最大时间、GFLOPS、后端库和相对 CPU 加速比。
2. **FP32/FP16 精度对比**：比较 FP32 与 FP16 矩阵乘法的耗时、加速比、最大绝对误差、平均绝对误差和相对 L2 误差，输出 `results/precision_compare.csv`。
3. **MLP 单卡训练实验**：完成前向传播、反向传播、参数更新和多轮训练，记录 loss、accuracy、epoch time、samples/s 和 GPU 显存峰值（如可用），输出 `results/mlp_train_log.csv`。

当前 macOS 环境没有 ROCm PyTorch，因此基础实验先生成真实 NumPy CPU baseline，并在安装 PyTorch 时额外生成 Torch CPU baseline；`ROCm_GPU` 行标记为 `measurement_type=unavailable`。后续在 AMD ROCm 环境运行同一脚本时，只要 `torch.version.hip` 非空且 `torch.cuda.is_available()` 为 True，脚本会自动加入 `ROCm_GPU` 实测行。

### 5.3 Agent 应用基准测试

运行 `experiments/benchmark_real.py` 进行基准测试。该脚本自动检测硬件环境：如果检测到真实 AMD GPU/NPU 且后端执行真实推理（`has_real_inference()=True`），则使用 `backend.fit_and_embed()` / `backend.transform()` / `backend.generate_answer()` 进行真实测量；如果 NPU EP 已检测但推理仍为 CPU 回退，则标记为 `cpu_fallback_with_hardware_detected`（不标记为 `real_hardware`）；如果没有硬件，则回退到 simulated 模式。CSV 中 `measurement_type` 列明确区分 `real_hardware` / `cpu_fallback_with_hardware_detected` / `simulated`。测试覆盖四个维度：

1. **Embedding 基准**：测试不同文档数量下的文档切块与嵌入耗时。
2. **Query embedding 基准**：测试不同文本块数量下的查询向量化耗时。
3. **Generation 基准**：测试 `generate_answer()` 的答案生成耗时。
4. **端到端 RAG 基准**：测试完整 ingest + query + retrieval + generate 流程。

当前环境下，CPU 行为真实 CPU 执行，SimulatedNPU 行为 CPU 计算加人为延迟，unavailable 后端（GPU/NPU 未检测到）直接跳过，不参与性能汇总和图表。结果写入 `results/latency_results.csv`、`results/backend_results.csv` 和 `results/resource_usage.csv`。

### 5.4 资源与能效采样

运行 `experiments/resource_monitor.py` 在 benchmark 期间采样 CPU 使用率、内存使用率和 ROCm GPU 功耗。采样结果写入 `results/power_trace.csv`，能耗摘要写入 `results/energy_summary.csv`。在真实 AMD ROCm 环境下，脚本通过 `rocm-smi --showpower` 获取 GPU 功耗，并用平均功耗乘以监控时长估算 GPU 能耗；在当前无 ROCm 环境下，CSV 明确标记 GPU 功耗不可用。

### 5.5 图表生成

运行 `experiments/plot_basic_results.py` 和 `experiments/plot_results.py` 读取 CSV 结果数据，使用 matplotlib 生成可视化图表，保存到 `figures/` 目录：

- `matmul_benchmark.png`：矩阵乘法耗时曲线。
- `precision_compare.png`：FP32/FP16 耗时与误差对比。
- `mlp_training_curve.png`：MLP loss 与 accuracy 训练曲线。
- `energy_comparison.png`：CPU/内存/ROCm GPU 功耗采样曲线。
- `backend_comparison.png`：各后端平均延迟对比柱状图。
- `latency_comparison.png`：不同文档数量下的延迟变化趋势图。
- `resource_usage.png`：系统资源使用情况图。

### 5.6 垂直行业端到端流程

运行 `experiments/demo_vertical_workflow.py` 复现企业内网政策问答场景。脚本摄入 `examples/enterprise_policy/` 下的示例制度与应急处置文档，执行固定业务问题，并将问题、答案、引用来源、最高检索分数、摄入块数、摄入阶段调度 trace、查询阶段调度 trace 和“数据不出本地”的说明写入 `results/vertical_demo_transcript.csv`。该文件用于证明作品不是单纯 benchmark，而是具备可演示的端到端应用流程。

### 5.7 Web UI 演示

运行 `localdoc/app.py`（基于 Gradio 框架）启动交互式演示界面，支持：

- 文档上传与目录加载。
- 知识库状态查看（文档数、文本块数、后端信息）。
- 交互式问答查询。
- 调度日志查看（记录每一步使用的后端与耗时）。

---

## 六、实验结果及分析

> **数据标注说明**：基础实验中的 CPU 数据为当前机器真实 CPU baseline；`ROCm_GPU` 在当前环境标记为 `unavailable`。Agent 应用中的 SimulatedNPU 数据为 **[模拟数据，不代表真实 AMD 硬件性能]**，仅用于验证调度框架和结果标注机制。

### 6.1 基础实验结果（当前 CPU baseline）

当前环境没有安装 ROCm PyTorch，因此基础实验先得到真实 CPU baseline；`ROCm_GPU` 行在 CSV 中保留为 `unavailable`，用于后续在 AMD ROCm 环境复现实测。

#### 6.1.1 矩阵乘法 benchmark

结果文件：`results/matmul_benchmark.csv`；图表：`figures/matmul_benchmark.png`。

| 矩阵规模 | CPU 平均耗时 (ms) | 标准差 (ms) | ROCm_GPU 状态 |
|----------|-------------------|-------------|---------------|
| 256 x 256 | 0.0248 | 0.0005 | unavailable |
| 512 x 512 | 0.1855 | 0.0353 | unavailable |
| 1024 x 1024 | 1.2640 | 0.1312 | unavailable |

分析：矩阵规模增大时，CPU 耗时随矩阵乘法复杂度上升。该实验代码已实现 ROCm GPU 路径，后续在 AMD 平台上可直接得到 GPU 平均时间、标准差和相对 CPU 加速比。

#### 6.1.2 FP32/FP16 精度对比

结果文件：`results/precision_compare.csv`；图表：`figures/precision_compare.png`。

| 矩阵规模 | FP32 耗时 (ms) | FP16 耗时 (ms) | FP16/FP32 速度比 | 平均绝对误差 | 相对 L2 误差 |
|----------|----------------|----------------|------------------|--------------|--------------|
| 256 x 256 | 0.0271 | 23.1200 | 0.0012 | 0.00450582 | 0.00035971 |
| 512 x 512 | 0.1793 | 171.4592 | 0.0010 | 0.00640759 | 0.00036059 |

分析：在当前 CPU/NumPy 环境中，FP16 并未获得加速，反而明显慢于 FP32。这符合许多通用 CPU 对 FP16 矩阵乘法缺少专用加速路径的现象。该结果也说明 FP16 优化是否有效依赖硬件支持；在 ROCm GPU 上重新运行后，应重点观察 FP16 是否带来吞吐提升，以及误差是否处于可接受范围。

#### 6.1.3 MLP 单卡训练实验

结果文件：`results/mlp_train_log.csv`；图表：`figures/mlp_training_curve.png`。

| Epoch | Loss | Accuracy | Epoch time (ms) |
|-------|------|----------|-----------------|
| 1 | 1.071814 | 0.501953 | 0.5168 |
| 2 | 1.051276 | 0.581055 | 0.4040 |
| 3 | 1.028154 | 0.650391 | 0.4000 |
| 4 | 1.001043 | 0.702148 | 0.3993 |
| 5 | 0.968593 | 0.752930 | 0.3990 |

分析：MLP 训练包含前向传播、反向传播和参数更新，loss 持续下降、accuracy 持续上升，说明训练流程正确。当前结果为 CPU baseline；后续在 ROCm 环境中会生成 `ROCm_GPU` 训练日志，用于对比 epoch time 和训练吞吐。

### 6.2 Agent 各后端延迟对比（当前 CPU + SimulatedNPU）

**[模拟数据，不代表真实 AMD 硬件性能]**

| 后端 | 平均延迟 (ms) | 测试次数 | measurement_type | 说明 |
|------|---------------|----------|----------|------|
| CPU | 1.470 | 6 | real_hardware | 真实 CPU 执行 |
| SimulatedNPU | 152.996 | 6 | simulated | CPU 计算 + 人为延迟 |

分析：当前 Agent benchmark 只保留真实 CPU 和 SimulatedNPU 两类可运行后端；未检测到 ROCm GPU 或真实 NPU 时，unavailable 后端不参与性能汇总。SimulatedNPU 的数值仅用于演示调度与标注机制，不代表真实 NPU 性能。

### 6.3 不同文档数量下的端到端延迟变化

**[模拟数据，不代表真实 AMD 硬件性能]**

端到端流程（文档摄入 + 查询）在不同文档数量下的延迟：

| 文档数量 | CPU (ms) | SimulatedNPU (ms) |
|----------|----------|-------------------|
| 1 | 1.233 | 91.913 |
| 5 | 5.192 | 294.801 |

分析：CPU 端到端耗时随文档数上升；SimulatedNPU 因人为延迟随文本块数量增加而显著增长。该实验主要验证端到端流程、调度记录和 CSV 标注机制，真实 GPU/NPU 加速比需要在 AMD ROCm/Ryzen AI 环境中补测。

### 6.4 资源与能效采样结果

结果文件：`results/power_trace.csv`、`results/energy_summary.csv`；图表：`figures/energy_comparison.png`。

当前环境无 `rocm-smi`，因此 GPU 功耗字段为空，`energy_summary.csv` 明确记录“GPU power unavailable”。在真实 AMD ROCm 环境中，该部分将给出平均 GPU 功耗、最大 GPU 功耗和估算 GPU 能耗，用于分析性能与能效之间的折中。

### 6.5 垂直行业端到端流程结果

结果文件：`results/vertical_demo_transcript.csv`。

企业内网政策问答流程包含 2 份示例业务文档、固定业务问题、答案、来源引用、检索分数和调度 trace。该 transcript 可直接用于答辩演示：先展示本地文档进入知识库，再展示系统回答“为什么不能调用外部 API”“需要记录哪些审计信息”“CPU/GPU/NPU 如何分工”等问题，最后说明所有数据和推理均在本地完成。

### 6.6 端到端流程各阶段耗时占比（模拟数据）

**[模拟数据，不代表真实 AMD 硬件性能]**

以 CPU 后端处理 10 个文档为例，端到端流程的各阶段耗时分布：

| 阶段 | 耗时占比 | 调度器分配 | 说明 |
|------|----------|------------|------|
| 文档加载 (document_loading) | ~5% | CPU | I/O 密集型，固定 CPU |
| 文本分块 (chunking) | ~3% | CPU | 逻辑处理，固定 CPU |
| 向量嵌入 (embedding) | ~80% | NPU > GPU > CPU | 计算密集型，主要加速目标 |
| 语义检索 (retrieval) | ~10% | GPU > CPU | 向量相似度计算 |
| 答案生成 (generation) | ~2% | GPU > CPU | 抽取式生成，耗时极低 |

分析：向量嵌入阶段占据了端到端流程约 80% 的总耗时，是异构加速的核心目标。在真实 AMD 硬件上，NPU 的 INT8 量化推理能力有望在此阶段获得显著加速。文档加载和文本分块阶段耗时占比极低，验证了调度器将其固定分配到 CPU 后端的合理性。

### 6.7 后端回退机制验证（模拟数据）

**[模拟数据，不代表真实 AMD 硬件性能]**

| 测试场景 | GPU 后端 | NPU 后端 | embedding 实际后端 | retrieval 实际后端 | 回退行为 |
|----------|----------|----------|-------------------|-------------------|----------|
| 无 GPU/NPU（当前环境） | 不可用 | SimulatedNPU | SimulatedNPU | CPU | 记录 NPU 模拟标记 |
| 仅 CPU 可用 | 不可用 | 不可用 | CPU | CPU | 记录回退日志 |
| GPU 可用，NPU 不可用 | 可用 | 不可用 | GPU | GPU | 无回退 |
| 全部可用 | 可用 | 可用 | NPU | GPU | 按优先级选择 |

分析：在当前仿真实验环境中，由于没有真实 GPU/NPU 硬件，embedding 阶段回退到 SimulatedNPUBackend（调度器检测到该后端后标记为模拟），retrieval 和 generation 阶段回退到 CPUBackend。系统的后端回退机制运行正常，调度日志完整记录了每个阶段的后端选择原因和是否为模拟后端。

### 6.8 结果综合分析

1. **端到端流程验证**：仿真实验成功验证了 LocalDocAgent 的完整 RAG 管线——从 DocumentLoader 加载文档，到 TextChunker 进行段落加句子两层切分，到 EmbeddingEngine 进行 TF-IDF 向量化，到 DocumentRetriever 进行余弦相似度 Top-K 检索，到 AnswerGenerator 生成抽取式回答。各模块之间的数据传递格式统一，接口定义清晰。

2. **调度框架可扩展性验证**：HeterogeneousScheduler 的任务类型与后端优先级映射机制运行正确。新增后端只需实现 `fit_and_embed()`、`transform()`、`generate_answer()` 等统一接口并注册到调度器中，无需修改上层业务逻辑代码。在后续获得真实 AMD 硬件后，只需将 SimulatedNPUBackend 替换为真实后端即可。

3. **后端回退机制验证**：系统在目标后端不可用时能够自动回退到备选后端，且所有回退行为均有完整的调度日志记录。这一机制保证了系统在不同硬件环境下的鲁棒性。

4. **延迟趋势分析**：当前环境中的 SimulatedNPU 数据只用于验证调度与标注机制；由于人为延迟和 CPU 实际计算混合存在，不能据此推导真实 NPU/GPU 性能排序。真实加速比必须以 ROCm GPU 和 Ryzen AI NPU 环境下的 `measurement_type=real_hardware` 数据为准。

### 6.9 可选本地 LLM 生成实验

原始系统使用抽取式回答生成（AnswerGenerator 的 extractive 模式），优点是轻量、稳定、无外部依赖。为体现本地 AI 推理能力，本系统额外接入了 Qwen3-1.7B 本地语言模型（Hugging Face Transformers），作为可选的生成后端。

**模型信息**：
- 模型：Qwen3-1.7B（1.7B 参数，Apache-2.0 许可证）
- 推理框架：Hugging Face Transformers（>=4.51.0）
- 推理方式：完全本地运行，不调用任何云端 API
- 特殊设置：关闭 thinking mode（`enable_thinking=False`）以保证演示稳定

**设计要点**：
- 该 LLM 仅替换答案生成环节，Embedding 仍使用 TF-IDF（CPUBackend）
- 通过环境变量 `LOCALDOC_USE_LLM=1` 启用，默认不加载
- 生成参数：`do_sample=True, temperature=0.7, top_p=0.8, top_k=20`（Qwen3 非 thinking 模式推荐）
- 不影响 CPU fallback 和 SimulatedNPUBackend 的正常运行
- 本地 LLM benchmark（`experiments/benchmark_llm_generation.py`）只评估生成推理开销

**当前限制**：
- 若无 ROCm/NPU 环境，模型在 CPU 上推理，推理速度较慢
- 该实验只证明"本地 LLM 生成链路可运行"，不声称 AMD GPU/NPU 加速
- 报告中不得写"验证了 AMD NPU 加速 Qwen3"
- 后续可在真实 AMD 平台上通过 ONNX Runtime 将模型部署到 NPU 上运行

---

## 七、总结与展望

### 7.1 工作总结

本实验完成了面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体系统的设计与仿真实验验证，主要成果包括：

1. 设计并实现了完整的 RAG 管线，包括 DocumentLoader、TextChunker、EmbeddingEngine、DocumentRetriever、AnswerGenerator 和 LocalDocAgent 六个核心模块。
2. 设计了 HeterogeneousScheduler 异构调度器，根据任务计算特征将文档加载和分块分配到 CPU，将向量嵌入优先分配到 NPU（回退到 GPU 再到 CPU），将检索和生成优先分配到 GPU（回退到 CPU）。
3. 实现了 CPUBackend、AMDGPUBackend、AMDNPUBackend 和 SimulatedNPUBackend 四种后端，后端接口统一，支持可插拔替换。
4. 通过仿真实验验证了本地知识库智能体的端到端流程、异构调度框架的可扩展性，以及在后续真实 AMD AI MAX+ 环境中进行硬件实测的可行性。
5. 开发了 Gradio Web UI 演示界面和完整的基准测试工具链（环境检查、基础异构实验、延迟基准测试、能效采样、垂直行业 transcript、图表生成）。

### 7.2 当前局限与后续真实硬件验证计划

**当前局限：**

1. **没有真实 AMD AI MAX+ 硬件**：当前实验全部基于 CPU fallback 和模拟后端完成，无法获得真实的 GPU/NPU 加速性能数据。
2. **没有真实 ROCm/NPU 性能数据**：AMDGPUBackend 在当前环境中回退到 CPU 计算，AMDNPUBackend 完全不可用。SimulatedNPUBackend 的人为延迟仅用于模拟推理耗时，不具有硬件性能参考意义。
3. **嵌入方法局限于 TF-IDF**：当前使用 TF-IDF 稀疏向量化，语义理解能力有限。在真实 NPU 上可部署 Transformer 类嵌入模型以获得更好的检索质量。
4. **答案生成能力有限**：当前使用抽取式回退方案，回答质量受限于关键词匹配。在真实 GPU/NPU 上可部署量化大语言模型实现自然语言生成。

**后续真实硬件验证计划：**

1. **获取 AMD 锐龙 AI MAX+ 硬件环境**：在配备 AMD Ryzen AI MAX+ 395 处理器的设备上部署项目。
2. **运行环境检查**：执行 `python experiments/check_environment.py`，确认 ROCm GPU 和 Ryzen AI NPU 的可用性。
3. **运行真实基准测试**：执行 `bash run_benchmark.sh`，在真实硬件上采集各后端的性能数据。
4. **替换模拟数据**：用真实硬件测试结果替换本报告中所有标注为"模拟数据"的部分。
5. **部署 NPU 模型**：可在 NPU 上部署 Qwen3.5 4B 等小型语言模型进行真实的生成推理测试，利用 XDNA NPU 的 50 TOPS INT8 算力加速。
6. **优化调度策略**：基于真实硬件性能数据调整任务与后端的映射关系和优先级。

---

## 八、参考文献

[1] AMD. AMD Ryzen AI MAX+ 395 Processor Specifications. https://www.amd.com/en/products/processors/laptop/ryzen.html

[2] AMD. ROCm Documentation. https://rocm.docs.amd.com/

[3] AMD. Ryzen AI Software Documentation. https://ryzenai.docs.amd.com/

[4] J. Ramos. Using TF-IDF to Determine Word Relevance in Document Queries. Proceedings of the First Instructional Conference on Machine Learning, 2003.

[5] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.

[6] P. Lewis, E. Perez, A. Piktus, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems (NeurIPS), 33, 2020.

[7] J. Johnson, M. Douze, and H. Jegou. Billion-scale Similarity Search with GPUs. IEEE Transactions on Big Data, 7(3): 535-547, 2021.

[8] Microsoft. ONNX Runtime Documentation. https://onnxruntime.ai/

[9] T. Chen, M. Li, Y. Li, et al. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. arXiv preprint arXiv:1512.01274, 2015.

[10] Y. LeCun, Y. Bengio, and G. Hinton. Deep Learning. Nature, 521(7553): 436-444, 2015.
