# ROCm 工具性能检测与调优记录

生成时间：2026-06-09T13:09:07

## 工具覆盖情况

- 已检测到的工具：rocminfo, amd-smi list, amd-smi static, amd-smi metric, rocm-smi, rocprofiler tools, rocprofv3 probe
- 未检测到或未安装的工具：rocm-bandwidth-test
- 命令非零退出或超时：rocprofiler tools, rocprofv3 probe

## 本项目如何使用 ROCm 工具

| 工具 | 本项目用途 | 产物 |
|------|------------|------|
| rocminfo | 枚举 ROCm GPU agent、gfx 架构、ROCm 栈是否工作 | `results/rocminfo.txt` |
| AMD SMI | GPU 型号、驱动、温度、功耗、显存和利用率监控 | `results/amd_smi_*.txt` |
| ROCm SMI | 兼容旧环境的 GPU 功耗、显存、频率、温度、利用率监控 | `results/rocm_smi_performance.txt` |
| ROCm Bandwidth Test | 测 CPU-GPU/GPU-GPU 数据传输带宽，判断数据搬运瓶颈 | `results/rocm_bandwidth_test.txt` |
| rocprofv3 / ROCProfiler | 采集 HIP runtime、kernel、memory copy trace，定位 kernel 与拷贝瓶颈 | `results/rocprofiler_tools.txt`、`results/rocprofiler_run.txt` |
| ROCm Compute/System Profiler | 用于进一步做 CU/L2/Speed-of-Light 分析和 CPU/GPU 系统级 trace | `results/rocprofiler_tools.txt` |

## 调优结论写法

1. 如果 `results/matmul_benchmark.csv` 中 `ROCm_GPU` 相比 CPU 有明显 speedup，报告中可说明矩阵密集型任务适合放到 GPU。
2. 如果 `results/precision_compare.csv` 中 FP16 速度更快且 `relative_l2_error` 可接受，报告中可说明半精度能提升吞吐，但需要结合误差约束使用。
3. 如果 `results/rocm_bandwidth_test.txt` 显示 CPU-GPU 带宽较低，报告中要把数据搬运列为瓶颈，并说明应减少 host/device 往返、批量化传输、复用 GPU resident tensor。
4. 如果 `results/amd_smi_metric.txt` 或 `results/rocm_smi_performance.txt` 中 GPU 利用率低，报告中可说明当前 RAG/LLM 小批量负载未充分填满 GPU，应通过 batch、增大矩阵规模或并发请求提升占用率。
5. 如果 `results/power_trace.csv` 有 GPU power 采样，报告中用 `energy_summary.csv` 说明性能-能耗折中；如果没有采样，明确写成工具不可用或容器权限不足。
6. 如果 `results/rocprofiler_run.txt` 产生 trace/summary 文件，报告中截取 kernel/runtime/memory copy 摘要，用它支撑“调优依据来自 ROCm profiler”。

## 面向本项目的优化建议

- 基础实验：扩大 matmul size 和 batch size，让 ROCm GPU 相比 CPU 的优势更明显。
- LLM 推理：限制 `max_new_tokens`，固定 prompt 长度，分别报告 prefill/generation 或总 tokens/s。
- RAG 管线：embedding/query 保持批处理，减少逐 chunk Python 循环；文档入库先统一切块再统一 fit/embed。
- 数据搬运：PyTorch tensor 创建后尽量保留在 GPU，避免每次 query 都在 CPU/GPU 间复制。
- 能效：同时报告 latency、tokens/s、GPU power、estimated energy，不只报告耗时。

## 截图建议

- 打开 `results/rocm_tools_summary.csv`，截工具 available、command、output_file。
- 打开 `results/amd_smi_metric.txt` 或 `results/rocm_smi_performance.txt`，截 GPU power/temp/utilization。
- 打开 `results/rocm_bandwidth_test.txt`，截 CPU-GPU bandwidth。
- 打开 `results/rocprofiler_run.txt`，截 rocprofv3 命令和生成文件列表。
