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

检测结论：当前环境为 "CPU fallback + simulated backend"，未检测到真实 AMD 硬件。

### 5.2 仿真实验基准测试

运行 `experiments/benchmark_latency.py` 进行模拟延迟基准测试。该脚本不依赖真实硬件，而是通过后端延迟乘数（CPU=1.0、GPU=0.6、NPU=0.3、SimulatedNPU=0.45）模拟不同后端策略的延迟差异。测试覆盖三个维度：

1. **文档摄入基准**：测试不同文档数量（1、5、10、20）在各模拟后端下的摄入延迟。
2. **查询基准**：测试不同文本块数量（10、50、100）在各模拟后端下的查询延迟。
3. **端到端基准**：测试完整摄入加查询管线在各配置下的总延迟。

所有数据标记为 `is_simulated=True`，`measurement_type="simulated_latency"`。结果写入 `results/latency_results.csv`、`results/backend_results.csv` 和 `results/resource_usage.csv`。

### 5.3 图表生成

运行 `experiments/plot_results.py` 读取 CSV 结果数据，使用 matplotlib 生成可视化图表，保存到 `figures/` 目录：

- `backend_comparison.png`：各后端平均延迟对比柱状图。
- `latency_comparison.png`：不同文档数量下的延迟变化趋势图。
- `resource_usage.png`：系统资源使用情况图。

### 5.4 Web UI 演示

运行 `localdoc/app.py`（基于 Gradio 框架）启动交互式演示界面，支持：

- 文档上传与目录加载。
- 知识库状态查看（文档数、文本块数、后端信息）。
- 交互式问答查询。
- 调度日志查看（记录每一步使用的后端与耗时）。

---

## 六、实验结果及分析

> **数据标注说明**：以下所有实验数据均为 **[模拟数据，不代表真实 AMD 硬件性能]**。数据来源于仿真实验的后端延迟乘数模型，用于验证调度框架的行为差异，而非衡量真实硬件性能。

### 6.1 各后端延迟对比（模拟数据）

**[模拟数据，不代表真实 AMD 硬件性能]**

| 后端 | 平均延迟 (ms) | 测试次数 | 延迟乘数 | 说明 |
|------|---------------|----------|----------|------|
| CPU | 40.571 | 11 | 1.0 | 真实 CPU 执行 |
| GPU | 24.508 | 11 | 0.6 | 模拟 GPU 加速比 |
| NPU | 12.447 | 11 | 0.3 | 模拟 NPU 加速比 |
| SimulatedNPU | 18.548 | 11 | 0.45 | CPU 计算 + 人为延迟 |

分析：延迟乘数模型反映了不同后端在理论上的性能差异趋势。NPU 模拟延迟最低，体现了 NPU 在低功耗推理场景下的设计优势；GPU 模拟延迟居中，体现了 GPU 并行计算的加速效果。需要强调的是，这些数据仅为调度框架行为验证的参考，不代表真实硬件性能。

### 6.2 不同文档数量下的端到端延迟变化（模拟数据）

**[模拟数据，不代表真实 AMD 硬件性能]**

端到端流程（文档摄入 + 查询）在不同文档数量下的延迟：

| 文档数量 | CPU (ms) | GPU (ms) | NPU (ms) | SimulatedNPU (ms) |
|----------|----------|----------|----------|-------------------|
| 1 | 6.387 | 3.709 | 1.916 | 2.869 |
| 5 | 31.094 | 18.951 | 9.505 | 14.174 |
| 10 | 60.695 | 37.384 | 18.968 | 28.186 |
| 20 | 123.787 | 74.841 | 37.687 | 56.283 |

分析：所有后端的延迟均与文档数量近似线性关系，说明系统的计算复杂度主要由文档摄入阶段（TF-IDF 计算）主导。各后端之间的加速比在不同文档数量下保持稳定，验证了调度框架在不同数据规模下的行为一致性。

### 6.3 端到端流程各阶段耗时占比（模拟数据）

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

### 6.4 后端回退机制验证（模拟数据）

**[模拟数据，不代表真实 AMD 硬件性能]**

| 测试场景 | GPU 后端 | NPU 后端 | embedding 实际后端 | retrieval 实际后端 | 回退行为 |
|----------|----------|----------|-------------------|-------------------|----------|
| 无 GPU/NPU（当前环境） | 不可用 | SimulatedNPU | SimulatedNPU | CPU | 记录 NPU 模拟标记 |
| 仅 CPU 可用 | 不可用 | 不可用 | CPU | CPU | 记录回退日志 |
| GPU 可用，NPU 不可用 | 可用 | 不可用 | GPU | GPU | 无回退 |
| 全部可用 | 可用 | 可用 | NPU | GPU | 按优先级选择 |

分析：在当前仿真实验环境中，由于没有真实 GPU/NPU 硬件，embedding 阶段回退到 SimulatedNPUBackend（调度器检测到该后端后标记为模拟），retrieval 和 generation 阶段回退到 CPUBackend。系统的后端回退机制运行正常，调度日志完整记录了每个阶段的后端选择原因和是否为模拟后端。

### 6.5 结果综合分析

1. **端到端流程验证**：仿真实验成功验证了 LocalDocAgent 的完整 RAG 管线——从 DocumentLoader 加载文档，到 TextChunker 进行段落加句子两层切分，到 EmbeddingEngine 进行 TF-IDF 向量化，到 DocumentRetriever 进行余弦相似度 Top-K 检索，到 AnswerGenerator 生成抽取式回答。各模块之间的数据传递格式统一，接口定义清晰。

2. **调度框架可扩展性验证**：HeterogeneousScheduler 的任务类型与后端优先级映射机制运行正确。新增后端只需实现对应的 `embed_texts()` 和 `generate_answer()` 接口并注册到调度器中，无需修改上层业务逻辑代码。在后续获得真实 AMD 硬件后，只需将 SimulatedNPUBackend 替换为真实后端即可。

3. **后端回退机制验证**：系统在目标后端不可用时能够自动回退到备选后端，且所有回退行为均有完整的调度日志记录。这一机制保证了系统在不同硬件环境下的鲁棒性。

4. **延迟趋势分析**：仿真实验中各后端之间的延迟差异趋势（NPU < GPU < CPU）与异构计算的理论预期一致，说明延迟乘数模型能够合理反映不同计算单元的性能差异方向。但需要再次强调，具体数值不具有硬件性能参考意义。

---

## 七、总结与展望

### 7.1 工作总结

本实验完成了面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体系统的设计与仿真实验验证，主要成果包括：

1. 设计并实现了完整的 RAG 管线，包括 DocumentLoader、TextChunker、EmbeddingEngine、DocumentRetriever、AnswerGenerator 和 LocalDocAgent 六个核心模块。
2. 设计了 HeterogeneousScheduler 异构调度器，根据任务计算特征将文档加载和分块分配到 CPU，将向量嵌入优先分配到 NPU（回退到 GPU 再到 CPU），将检索和生成优先分配到 GPU（回退到 CPU）。
3. 实现了 CPUBackend、AMDGPUBackend、AMDNPUBackend 和 SimulatedNPUBackend 四种后端，后端接口统一，支持可插拔替换。
4. 通过仿真实验验证了本地知识库智能体的端到端流程、异构调度框架的可扩展性，以及在后续真实 AMD AI MAX+ 环境中进行硬件实测的可行性。
5. 开发了 Gradio Web UI 演示界面和完整的基准测试工具链（环境检查、延迟基准测试、图表生成）。

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
