# 《异构计算》实验报告

**题目：基于 AMD Radeon 8060S（Strix Halo iGPU）ROCm 平台的本地知识库智能体与异构资源调度实验**

## 摘要
本实验围绕 AMD 锐龙 AI MAX+ / Strix Halo 平台的端侧 AI 应用展开，设计并实现 LocalDoc Agent 本地知识库智能体。系统覆盖文档加载、文本切块、TF-IDF/ROCm 后端嵌入、语义检索、抽取式回答、本地 Qwen3-1.7B LLM 生成和 Gradio Web 演示，强调数据不出机、不调用外部 API 的隐私敏感场景。AMD 平台实测显示：PyTorch ROCm/HIP 可用，Radeon 8060S Graphics 可执行 tensor probe，基础实验生成 CPU/Torch CPU/ROCm GPU 对比数据，Qwen3-1.7B 通过 ROCm GPU 运行在 device=cuda（PyTorch HIP 兼容接口）。NPU 在本环境未检测到，SimulatedNPU 仅用于调度教学演示，不作为真实 NPU 性能。

**关键词：** 异构计算、ROCm、PyTorch、Radeon 8060S、端侧 AI、RAG、本地 LLM、Qwen3

## 1 实验目的与要求
1. 验证 AMD ROCm GPU 环境，包括 rocminfo、rocm-smi、PyTorch HIP、GPU 名称、gfx 架构和 tensor 执行能力。
2. 完成矩阵乘法、FP32/FP16 精度对比、MLP 单卡训练三类基础异构实验。
3. 构建可演示的端侧本地知识库智能体，体现本地推理、端到端应用价值和隐私数据不出机。
4. 使用 AMD SMI、ROCm SMI、ROCProfiler 相关工具采集性能检测与调优证据。
5. 对 CPU/GPU 性能、半精度收益、数据搬运与当前局限进行分析。

## 2 实验环境
- Python：3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]；可执行文件：/usr/bin/python3
- 平台：Linux-6.14.0-1018-oem-x86_64-with-glibc2.39；内核：6.14.0-1018-oem
- CPU 线程数：32；内存：64069 MB
- PyTorch：2.9.1+rocm7.12.0；HIP：7.12.60610-2bd1678d3d；torch.cuda.is_available()：True
- GPU：Radeon 8060S Graphics；GPU 架构：(11, 5)；gfx 架构：['gfx11', 'gfx1151']
- ROCm tensor probe ok：True；当前模式：real hardware execution
- NPU：False。本机未检测到 Ryzen AI / ONNX Runtime NPU Execution Provider。
- ROCm SMI 采样证据：温度 26.0 C，功耗 23.08 W，GPU use 1%，VRAM total 68719476736 B，used 154816512 B。

## 3 运行命令
```bash
# AMD ROCm 平台一键复现实验
LOCALDOC_USE_CURRENT_PYTHON=1 bash scripts/setup_llm.sh --skip-torch
python scripts/test_llm.py --require-gpu
bash run_all_experiments.sh --allow-llm-hub --require-llm-gpu

# 启动本地 Web 演示，默认启用 Qwen3-1.7B、要求 ROCm GPU、share=True
bash scripts/run_demo_llm.sh
```

## 4 系统设计与实现
LocalDoc Agent 采用 RAG（Retrieval-Augmented Generation）管线，完整流程为：文档加载 -> 文本切块 -> 向量嵌入 -> 索引存储 -> 查询检索 -> 答案生成。该流程能对应真实企业内网知识库场景，输入是本地制度、应急预案或业务文档，输出是带引用来源、检索分数和调度记录的问答结果。

核心模块包括：
- DocumentLoader：读取 Markdown、TXT、PDF 等本地文档，属于 I/O 密集任务，由 CPU 执行。
- TextChunker：按段落和句子进行文本切块，保留重叠窗口，避免上下文断裂，属于 CPU 逻辑处理任务。
- EmbeddingEngine：构建文档向量和查询向量。CPU baseline 使用 TF-IDF；GPU 后端通过 PyTorch HIP 执行可迁移到 ROCm 的张量运算。
- DocumentRetriever：基于余弦相似度返回 Top-K 文档块，是可并行化的向量检索阶段。
- AnswerGenerator：支持抽取式回答和本地 LLM 生成。本地 LLM 模式使用 Qwen3-1.7B，不调用外部 API。
- HeterogeneousScheduler：根据任务类型选择 CPU/GPU/NPU/SimulatedNPU 后端，并在日志中记录 selected_backend、耗时和 simulated 标志。

调度策略为：文档加载和文本切块固定使用 CPU；嵌入阶段优先 NPU，回退 GPU/CPU；检索和生成优先 GPU，回退 CPU。当前 AMD 8060S 环境没有检测到真实 NPU，因此 GPU 与 CPU 构成真实硬件对比，SimulatedNPU 只用于证明调度框架能识别和标注模拟后端。

本地 LLM 后端 LocalLLMBackend 使用 Qwen3-1.7B。设置 LOCALDOC_REQUIRE_LLM_GPU=1 后，如果 ROCm GPU 不可用会直接失败，避免将 CPU fallback 误写成 GPU 实测。实测记录中 device=cuda 是 PyTorch 对 HIP/ROCm GPU 沿用的统一接口名称，不表示使用 NVIDIA CUDA。

## 5 实验内容与步骤
实验按“环境确认 -> ROCm 工具检测 -> 基础实验 -> Agent 应用 benchmark -> 本地 LLM -> 图表生成”的顺序执行。

1. 环境检查：运行 experiments/check_environment.py，生成 environment_report.txt，记录 Python、Linux kernel、PyTorch 版本、HIP 版本、torch.cuda.is_available()、GPU 名称、gfx 架构、ROCm tensor probe 和 NPU EP 状态。
2. ROCm 工具检测：运行 experiments/rocm_tools_profile.py，采集 rocminfo、AMD SMI、ROCm SMI、Bandwidth Test、rocprofv3/ROCProfiler 等工具证据，并生成 rocm_tools_summary.csv 和 rocm_tuning_recommendations.md。
3. 基础异构实验：运行 experiments/basic_benchmarks.py，完成矩阵乘法、FP32/FP16 精度对比和 MLP 单卡训练，输出 matmul_benchmark.csv、precision_compare.csv 和 mlp_train_log.csv。
4. Agent 后端 benchmark：运行 experiments/benchmark_real.py，对 embedding、query_embedding、generation、end_to_end_rag 四类任务进行 CPU/GPU/SimulatedNPU 对比，输出 latency_results.csv 和 backend_results.csv。
5. 垂直行业应用：运行 experiments/demo_vertical_workflow.py，使用企业内网政策问答示例，输出 vertical_demo_transcript.csv。
6. 本地 LLM benchmark：运行 scripts/run_llm_benchmark.sh 或 run_all_experiments.sh 的 LLM 阶段，测试 Qwen3-1.7B 本地生成延迟、tokens/s、模型加载时间和 ROCm GPU 状态。
7. 图表生成：运行 plot_basic_results.py、plot_results.py 和 plot_llm_results.py，输出 figures/ 下的柱状图、折线图和训练曲线。

## 6 与评分表逐项对应
- 主题相关性：CPU/GPU、ROCm、PyTorch、FP32/FP16、MLP、RAG 本地智能体均已覆盖。状态：满足。
- 平台与任务合理性：AMD Radeon 8060S / gfx1151 / HIP 7.12.60610-2bd1678d3d；任务为本地知识库 RAG 与 LLM 推理。状态：满足。
- 环境检查：environment_report.txt、rocminfo.txt、rocm_smi_performance.txt、hipconfig_full.txt。状态：满足。
- 矩阵乘法 benchmark：matmul_benchmark.csv + matmul_benchmark.png；含 CPU/Torch_CPU/ROCm_GPU。状态：满足。
- FP32/FP16 对比：precision_compare.csv + precision_compare.png；含速度与误差。状态：满足。
- MLP 训练：mlp_train_log.csv + mlp_training_curve.png；含 forward/backward/update 多轮训练。状态：满足。
- 实际应用：enterprise_intranet_policy_qa transcript + Gradio Web demo + Qwen3-1.7B 本地推理。状态：满足。
- 异构适配：调度器、CPU/GPU/NPU/SimulatedNPU 后端、ROCm GPU LLM、measurement_type 标注。状态：满足。
- 性能与能效：延迟、GFLOPS、tokens/s、ROCm SMI 功耗记录；能耗时间序列解析已修复需补跑更新。状态：基本满足。
- 工具调优：AMD SMI/ROCm SMI/rocprofv3 工具探测；rocprofv3 旧结果失败，命令已修复需补跑。状态：基本满足。

## 7 实验结果摘要
### 7.1 矩阵乘法
- size=256 backend=CPU type=real_cpu avg_ms=0.1686 std_ms=0.0315 GFLOPS=198.96 speedup=1.00
- size=256 backend=Torch_CPU type=real_torch_cpu avg_ms=0.0966 std_ms=0.0350 GFLOPS=347.51 speedup=1.75
- size=256 backend=ROCm_GPU type=real_rocm_gpu avg_ms=0.1162 std_ms=0.1135 GFLOPS=288.80 speedup=0.83
- size=512 backend=CPU type=real_cpu avg_ms=2.9960 std_ms=0.0199 GFLOPS=89.60 speedup=1.00
- size=512 backend=Torch_CPU type=real_torch_cpu avg_ms=65.8259 std_ms=37.5129 GFLOPS=4.08 speedup=0.05
- size=512 backend=ROCm_GPU type=real_rocm_gpu avg_ms=0.2536 std_ms=0.0234 GFLOPS=1058.62 speedup=259.59
- size=1024 backend=CPU type=real_cpu avg_ms=32.3618 std_ms=41.3197 GFLOPS=66.36 speedup=1.00
- size=1024 backend=Torch_CPU type=real_torch_cpu avg_ms=32.3620 std_ms=41.2336 GFLOPS=66.36 speedup=1.00
- size=1024 backend=ROCm_GPU type=real_rocm_gpu avg_ms=0.8100 std_ms=0.0934 GFLOPS=2651.25 speedup=39.95

分析：小规模矩阵中 GPU 的启动和调度开销会抵消并行收益，因此 256 规模下 ROCm GPU 不一定领先；规模扩大后，GPU 在 512 和 1024 规模上体现出明显吞吐优势，说明矩阵密集型任务适合使用 GPU。

### 7.2 FP32/FP16
- size=256 backend=CPU type=real_cpu fp32_ms=0.0848 fp16_ms=39.2918 fp16_speedup=0.002 mean_abs_error=0.004506 rel_l2=0.000360
- size=256 backend=Torch_CPU type=real_torch_cpu fp32_ms=0.0422 fp16_ms=10.5288 fp16_speedup=0.004 mean_abs_error=0.004507 rel_l2=0.000360
- size=256 backend=ROCm_GPU type=real_rocm_gpu fp32_ms=0.0702 fp16_ms=0.1076 fp16_speedup=0.653 mean_abs_error=0.004466 rel_l2=0.000358
- size=512 backend=CPU type=real_cpu fp32_ms=3.3163 fp16_ms=315.7529 fp16_speedup=0.011 mean_abs_error=0.006408 rel_l2=0.000361
- size=512 backend=Torch_CPU type=real_torch_cpu fp32_ms=0.8245 fp16_ms=83.2358 fp16_speedup=0.010 mean_abs_error=0.006407 rel_l2=0.000361
- size=512 backend=ROCm_GPU type=real_rocm_gpu fp32_ms=0.1939 fp16_ms=0.0281 fp16_speedup=6.913 mean_abs_error=0.006388 rel_l2=0.000360

分析：CPU 上 FP16 没有加速，说明半精度优化必须依赖硬件支持。ROCm GPU 在 512 规模下 FP16 相比 FP32 获得明显速度收益，同时 relative L2 error 保持在约 3.6e-4，适合误差可控的推理或训练场景。

### 7.3 MLP
- backend=CPU type=real_cpu epoch=5 loss=0.968593 accuracy=0.7529 epoch_time_ms=0.301 samples/s=3401843.8 max_mem_mb=N/A
- backend=Torch_CPU type=real_torch_cpu epoch=5 loss=0.875841 accuracy=0.7373 epoch_time_ms=3.254 samples/s=314654.5 max_mem_mb=N/A
- backend=ROCm_GPU type=real_rocm_gpu epoch=5 loss=0.875732 accuracy=0.7275 epoch_time_ms=4.529 samples/s=226085.0 max_mem_mb=140.68

分析：MLP 覆盖前向传播、反向传播和参数更新。由于样本量较小，GPU 首轮和小 batch 开销明显，不能简单用该小数据结果推出所有训练任务 GPU 都更快；但 loss 下降、accuracy 上升证明训练流程完整可运行。

### 7.4 本地 LLM
- Qwen3-1.7B 平均 generation_time_s=8.401，平均 tokens/s=32.03，设备为 ROCm GPU 的 device=cuda。
- query=1 device=cuda hip=7.12.60610-2bd1678d3d probe=True load_s=14.61 gen_s=21.577 tokens/s=8.50 output_tokens=183
- query=2 device=cuda hip=7.12.60610-2bd1678d3d probe=True load_s=14.61 gen_s=1.676 tokens/s=43.00 output_tokens=72
- query=3 device=cuda hip=7.12.60610-2bd1678d3d probe=True load_s=14.61 gen_s=1.949 tokens/s=44.60 output_tokens=87

分析：首次查询包含模型预热和较长输出，耗时显著高于后续短查询。所有 LLM 记录均显示 device=cuda、HIP 版本非空、ROCm tensor probe 为 True，说明本地 Qwen3-1.7B 在 AMD ROCm GPU 上运行。该实验体现“数据不出机”的本地 AI 推理要求。

### 7.5 Agent 应用与资源结果
backend_results.csv 显示 GPU 和 CPU 均为 real_hardware，SimulatedNPU 明确标记为 simulated。vertical_demo_transcript.csv 记录了企业内网政策问答的用户问题、回答、引用来源、检索分数、延迟和 privacy_note，可用于报告和答辩截图。

ROCm SMI 记录了 GPU 当前功耗 23.08 W、温度 26.0 C、GPU use 1% 和 VRAM 使用情况。由于旧版 resource_monitor.py 没有识别 Power (W): value 格式，energy_summary.csv 没有生成时间序列功耗积分；该问题已经在代码中修复，若能再获得 AMD 环境，补跑后可获得 avg_gpu_power_w、max_gpu_power_w 和 estimated_gpu_energy_j。

## 8 运行中问题与处理
- RAG LLM 元数据：问题：rag_mode_comparison.csv 旧结果 device=cuda 但 torch_cuda_available=False；处理：benchmark_rag_modes.py 补写 torch_cuda_available、ROCm probe 字段；已修正当前 CSV。
- ROCm SMI 功耗采样：问题：resource_monitor.py 未识别 Power (W): 23.08 格式，energy_summary 无功耗样本；处理：新增 Power (W): value 与 value W 双格式解析，并加测试。
- rocprofv3 probe：问题：rocprofv3 新版命令要求 profiler 参数后加 -- 再跟应用程序；处理：rocm_tools_profile.py 增加 --，并加命令构造测试。

## 9 总结与展望
项目代码和已提交 AMD 结果已经覆盖评分表中除最终人工排版截图之外的大部分要求。真实 ROCm GPU 运行、本地 Qwen3-1.7B GPU 推理、基础 benchmark、实际应用流程和图表证据均已具备。当前主要扣分风险是能耗时间序列和 rocprofv3 probe 的旧结果需要在 AMD 平台按修复后代码补跑；如果无法补跑，报告中应以“问题处理与限制”方式说明，不能将旧失败结果写成成功。

后续工作包括：接入真实 Ryzen AI NPU / ONNX Runtime EP，替代 SimulatedNPU；补充 rocprofv3 trace、CPU-GPU 带宽测试和能耗积分结果；扩大文档集、矩阵规模和并发查询数，以进一步评估 GPU 在端侧 RAG 智能体中的吞吐、延迟和能效收益。
