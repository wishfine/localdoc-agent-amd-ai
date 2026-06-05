# 实验截图清单

本文档用于最后写报告和答辩时截图。建议截图文件名按 `01_环境检查.png`、`02_rocm_smi.png` 这种顺序保存，方便放入 Word。

## 一、必截

1. [必截] 一键全量实验终端结束页
   - 命令：`bash run_all_experiments.sh`
   - 截图内容：终端显示“全量实验完成”、`results/full_experiment_run.log`、`results/experiment_manifest.txt`、生成的 CSV 和图表列表。

2. [必截] 环境检查报告
   - 文件：`results/environment_report.txt`
   - 截图内容：Python、平台、PyTorch、`torch.version.hip`、`torch.cuda.is_available()`、ROCm GPU available、AMD NPU available。
   - 在 AMD 真机上重点截：GPU 名称、HIP 版本、gfx 架构。

3. [必截] ROCm 原始证据
   - 文件：`results/rocminfo.txt`、`results/rocm_smi.txt`、`results/hipcc_version.txt`、`results/hipconfig_full.txt`
   - 截图内容：`rocminfo` 中的 `gfx...`，`rocm-smi` 中的 GPU 名称/功耗/显存，`hipcc` 或 `hipconfig` 中的 ROCm/HIP 版本。
   - 当前 macOS 环境会显示 `COMMAND NOT FOUND`，这只能作为开发环境说明；最终 AMD 机器上要重新截图。

4. [必截] 矩阵乘法结果图
   - 文件：`figures/matmul_benchmark.png`
   - 同时打开：`results/matmul_benchmark.csv`
   - 截图内容：CPU/Torch CPU/ROCm GPU 的平均耗时、标准差、GFLOPS、speedup。

5. [必截] FP32/FP16 对比图
   - 文件：`figures/precision_compare.png`
   - 同时打开：`results/precision_compare.csv`
   - 截图内容：FP32/FP16 耗时、speedup、最大/平均误差、relative L2 error。

6. [必截] MLP 训练曲线
   - 文件：`figures/mlp_training_curve.png`
   - 同时打开：`results/mlp_train_log.csv`
   - 截图内容：loss 下降、accuracy 上升、epoch time、samples/s。

7. [必截] Agent 后端对比图
   - 文件：`figures/backend_comparison.png`
   - 同时打开：`results/backend_results.csv`
   - 截图内容：`measurement_type`、`is_simulated`、`real_inference`，证明没有把模拟数据写成真实硬件数据。

8. [必截] Agent 延迟趋势图
   - 文件：`figures/latency_comparison.png`
   - 同时打开：`results/latency_results.csv`
   - 截图内容：embedding、query embedding、generation、e2e_rag 四类 benchmark。

9. [必截] 能耗/资源图
   - 文件：`figures/energy_comparison.png`
   - 同时打开：`results/energy_summary.csv`
   - 截图内容：CPU/内存使用率；AMD ROCm 真机上还要截 GPU power 和 estimated energy。

10. [必截] 垂直行业端到端 transcript
    - 文件：`results/vertical_demo_transcript.csv`
    - 截图内容：企业内网政策问答的问题、答案、来源、top_score、`ingest_backend_trace`、`query_backend_trace`、`privacy_note`。

## 二、建议补截

11. [建议] Gradio Web UI 演示页
    - 命令：`bash run_demo.sh`
    - 截图内容：上传文档、构建知识库、输入问题、返回答案、调度日志。

12. [建议] 项目文件结构
    - 截图内容：`localdoc/`、`experiments/`、`examples/enterprise_policy/`、`results/`、`figures/`。
    - 用于说明代码完整性和实验产物组织。

13. [建议] 本地 LLM benchmark
    - 文件：`results/llm_generation_benchmark.csv`
    - 截图内容：如果本地模型未下载，截 skipped 记录；如果已下载，截模型路径、device、latency、tokens/s。

14. [建议] 单元测试结果
    - 命令：`python -m pytest -q`
    - 截图内容：`48 passed` 或当前实际通过数量。

## 三、放进报告时的顺序

1. 环境与硬件证据：环境报告、ROCm 原始证据。
2. 基础异构实验：matmul、FP32/FP16、MLP。
3. 应用实验：Agent latency、backend comparison、vertical transcript。
4. 性能与能效：resource usage、energy summary。
5. 诚信说明：`measurement_type`、`is_simulated`、`real_inference` 字段截图。
