"""
LocalDoc Agent - Gradio Web Demo

提供本地知识库智能体的 Web 界面。
支持文档上传、智能查询（含调度日志）、系统信息查看和基准测试。

设计为在 CPU 上即可运行，可选开启模拟 NPU 演示模式。
可选接入本地 LLM (Qwen3-1.7B)，通过 LOCALDOC_USE_LLM=1 环境变量启用。
所有 simulated backend 结果仅用于验证调度流程，不代表真实硬件性能。
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _build_agent(simulate_npu: bool = False):
    """Create a LocalDocAgent with scheduler, optional LLM, and optional simulated NPU."""
    from localdoc.agent import LocalDocAgent
    from localdoc.scheduler import HeterogeneousScheduler
    from localdoc.backends.cpu_backend import CPUBackend

    # Single shared CPUBackend instance for scheduler and agent
    cpu = CPUBackend()
    backends = {"cpu": cpu}
    backend = cpu  # default backend for agent

    # Check if local LLM is enabled via environment variable
    use_llm = os.getenv("LOCALDOC_USE_LLM", "0") == "1"
    if use_llm:
        try:
            from localdoc.backends.local_llm_backend import LocalLLMBackend
            llm_backend = LocalLLMBackend()
            if llm_backend.is_available():
                backend = llm_backend
                logger.info("Local LLM backend enabled: %s", llm_backend.name)
            else:
                logger.warning(
                    "LOCALDOC_USE_LLM=1 but model not found at %s. "
                    "Run: bash scripts/download_llm.sh",
                    llm_backend.model_path,
                )
        except Exception as e:
            logger.warning("Failed to load LocalLLMBackend: %s. Using CPU fallback.", e)

    if simulate_npu:
        try:
            from localdoc.backends.simulated_npu import SimulatedNPUBackend
            backends["npu"] = SimulatedNPUBackend()
        except ImportError:
            logger.warning("SimulatedNPUBackend 不可用，回退到 CPU")

    scheduler = HeterogeneousScheduler(backends=backends)
    return LocalDocAgent(backend=backend, scheduler=scheduler)


def _get_backend_status_report() -> str:
    """Check all backends and return a formatted status report."""
    lines = []

    # CPU backend
    from localdoc.backends.cpu_backend import CPUBackend
    cpu = CPUBackend()
    lines.append(f"| CPUBackend | ✅ real, always available | {cpu.name} |")

    # GPU backend
    try:
        from localdoc.backends.gpu_backend import AMDGPUBackend
        gpu = AMDGPUBackend()
        if gpu.is_available():
            info = gpu.get_device_info()
            lines.append(f"| AMDGPUBackend | ✅ real hardware detected | HIP {info.get('hip_version', '?')}, {info.get('gpu_name', '?')} |")
        else:
            lines.append("| AMDGPUBackend | ⚠️ unavailable (requires ROCm + PyTorch HIP) | — |")
    except Exception:
        lines.append("| AMDGPUBackend | ⚠️ import failed | — |")

    # NPU backend
    try:
        from localdoc.backends.npu_backend import AMDNPUBackend
        npu = AMDNPUBackend()
        if npu.is_available():
            info = npu.get_device_info()
            lines.append(f"| AMDNPUBackend | ✅ real hardware detected | EP: {info.get('execution_provider', '?')} |")
        else:
            lines.append("| AMDNPUBackend | ⚠️ unavailable (requires Ryzen AI SDK / ONNX EP) | — |")
    except Exception:
        lines.append("| AMDNPUBackend | ⚠️ import failed | — |")

    # Simulated NPU
    lines.append("| SimulatedNPUBackend | 🎭 simulated only (demo) | NOT real hardware |")

    header = "| Backend | Status | Detail |\n|---|---|---|\n"
    return header + "\n".join(lines)


def _format_schedule_report(report: Dict[str, Dict[str, str]]) -> str:
    """Format the scheduler report as markdown table."""
    header = "| 任务类型 | 分配后端 | 原因 | 模拟? |\n|---|---|---|---|\n"
    rows = []
    for task_type, info in report.items():
        sim_flag = "⚠️ 模拟" if info.get("is_simulated") else "—"
        rows.append(
            f"| `{task_type}` | **{info['backend']}** | {info['reason']} | {sim_flag} |"
        )
    return header + "\n".join(rows)


def _format_backend_trace(trace: List[Dict[str, Any]]) -> str:
    """Format backend trace from query result."""
    if not trace:
        return ""
    lines = ["\n\n**调度日志 / Backend Trace:**\n"]
    for entry in trace:
        sim = " [模拟]" if entry.get("is_simulated") else ""
        lines.append(
            f"- `{entry['task_type']}` → **{entry['backend']}**"
            f" ({entry['elapsed_seconds']:.4f}s){sim}"
        )
    return "\n".join(lines)


def create_app():
    """
    Create and return the Gradio Blocks application.
    Uses lazy import so the module can be loaded without gradio installed.
    """
    import gradio as gr

    _agent = None
    _ingested_count = 0

    def _ensure_agent(simulate_npu: bool):
        nonlocal _agent
        _agent = _build_agent(simulate_npu=simulate_npu)
        return _agent

    def handle_upload(files, simulate_npu: bool):
        if not files:
            return "未选择任何文件。"

        agent = _ensure_agent(simulate_npu)
        statuses = []
        nonlocal _ingested_count
        _ingested_count = 0

        for file in files:
            try:
                file_path = file.name if hasattr(file, "name") else str(file)
                chunk_count = agent.ingest_document(file_path)
                _ingested_count += chunk_count
                statuses.append(f"✅ {Path(file_path).name}: {chunk_count} 个文本块")
            except Exception as e:
                statuses.append(f"❌ {Path(file_path).name}: {e}")

        stats = agent.get_stats()
        statuses.append(f"\n📊 总计: {stats['chunk_count']} 个文本块")
        statuses.append(f"🔧 后端: {stats['backend']}")
        statuses.append(f"📋 调度器: {'已启用' if stats['scheduler_enabled'] else '未启用'}")

        return "\n".join(statuses)

    def handle_query(question: str, simulate_npu: bool):
        if not question.strip():
            return "请输入问题。", ""

        if _agent is None or _ingested_count == 0:
            return "请先上传并加载文档。", ""

        try:
            result = _agent.query(question, top_k=3)
            answer = result["answer"]
            sources = "\n".join(
                f"- {src}" for src in result["sources"]
            ) if result["sources"] else "无来源信息"

            latency_info = f"\n\n---\n⏱️ 查询耗时: {result['latency']:.3f}s"

            # Append backend trace if available
            trace_info = ""
            if "backend_trace" in result:
                trace_info = _format_backend_trace(result["backend_trace"])

            return answer + latency_info + trace_info, sources

        except Exception as e:
            return f"查询出错: {e}", ""

    def get_system_info(simulate_npu: bool):
        backend_status = _get_backend_status_report()

        # Use the agent's scheduler if available, otherwise create a fresh one
        if _agent is not None and _agent.scheduler is not None:
            schedule_report = _format_schedule_report(
                _agent.scheduler.get_schedule_report()
            )
        else:
            from localdoc.scheduler import HeterogeneousScheduler
            from localdoc.backends.cpu_backend import CPUBackend
            backends = {"cpu": CPUBackend()}
            if simulate_npu:
                try:
                    from localdoc.backends.simulated_npu import SimulatedNPUBackend
                    backends["npu"] = SimulatedNPUBackend()
                except ImportError:
                    pass
            scheduler = HeterogeneousScheduler(backends=backends)
            schedule_report = _format_schedule_report(scheduler.get_schedule_report())

        agent_stats = ""
        if _agent is not None:
            stats = _agent.get_stats()
            agent_stats = (
                f"\n\n### 智能体状态\n"
                f"- 已加载文档: {stats['document_count']}\n"
                f"- 文本块数量: {stats['chunk_count']}\n"
                f"- 后端: {stats['backend']}\n"
                f"- 调度器: {'已启用' if stats['scheduler_enabled'] else '未启用'}\n"
                f"- 可用后端: {stats['available_backends']}"
            )

        return backend_status, schedule_report + agent_stats

    def run_benchmark(simulate_npu: bool):
        from localdoc.scheduler import HeterogeneousScheduler, BenchmarkTaskType
        from localdoc.backends.cpu_backend import CPUBackend

        backends = {"cpu": CPUBackend()}
        if simulate_npu:
            try:
                from localdoc.backends.simulated_npu import SimulatedNPUBackend
                backends["npu"] = SimulatedNPUBackend()
            except ImportError:
                pass

        scheduler = HeterogeneousScheduler(backends=backends)

        lines = ["## ⚡ 基准测试结果 (Simulated Backend Policy)\n"]
        lines.append(f"**后端**: {list(scheduler.backends.keys())}\n")
        lines.append("| 任务类型 | 分配后端 | 耗时 (s) | 模拟? |")
        lines.append("|---|---|---|---|")

        for task_type in BenchmarkTaskType:
            scheduler.execute(task_type, lambda: time.sleep(0.001))

        log = scheduler.get_execution_log()
        for entry in log:
            sim = "⚠️ 模拟" if entry.get("is_simulated") else "—"
            lines.append(
                f"| `{entry['task_type']}` | {entry['backend']} | "
                f"{entry['elapsed_seconds']:.4f} | {sim} |"
            )

        lines.append(
            "\n> ⚠️ **注意**: 以上为 simulated backend policy 的结果，"
            "仅用于验证调度流程，**不可作为真实 AMD AI MAX+ 硬件性能结果**。"
        )
        return "\n".join(lines)

    # ---- Build the UI ----

    with gr.Blocks(
        title="LocalDoc Agent - 本地知识库智能体",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            """
# 📚 LocalDoc Agent - 本地知识库智能体 (AMD AI MAX+)

> **异构计算课程项目** | 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体原型
>
> ⚠️ 当前默认运行在 **CPU fallback 模式**。SimulatedNPUBackend 仅用于演示调度逻辑，
> **不代表真实 AMD NPU 性能**。
>
> 💡 可选：设置 `LOCALDOC_USE_LLM=1` 后启用本地 Qwen3-1.7B 生成后端。
> 该后端完全本地运行，不调用云端 API。
            """
        )

        with gr.Row():
            simulate_npu_toggle = gr.Checkbox(
                label="🎭 模拟 NPU 演示 (SimulatedNPUBackend)",
                value=False,
                info="开启后使用 SimulatedNPUBackend 模拟 NPU 调度（仅用于演示调度流程）",
            )

        with gr.Tabs():

            # Tab 1: Document Upload
            with gr.TabItem("📄 文档上传"):
                gr.Markdown("上传 `.md`、`.txt` 或 `.pdf` 文件构建本地知识库。")
                file_upload = gr.File(
                    label="选择文件",
                    file_count="multiple",
                    file_types=[".md", ".txt", ".pdf"],
                )
                upload_btn = gr.Button("📥 加载文档", variant="primary")
                upload_status = gr.Textbox(label="加载状态", lines=8, interactive=False)
                upload_btn.click(
                    fn=handle_upload,
                    inputs=[file_upload, simulate_npu_toggle],
                    outputs=[upload_status],
                )

            # Tab 2: Query
            with gr.TabItem("🔍 知识库查询"):
                gr.Markdown("输入问题，基于已加载的文档获取回答。查询结果包含调度日志。")
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：什么是异构计算？",
                    lines=2,
                )
                query_btn = gr.Button("🔎 提交查询", variant="primary")
                with gr.Row():
                    answer_output = gr.Textbox(label="回答", lines=10, interactive=False)
                    sources_output = gr.Textbox(label="引用来源", lines=10, interactive=False)
                query_btn.click(
                    fn=handle_query,
                    inputs=[question_input, simulate_npu_toggle],
                    outputs=[answer_output, sources_output],
                )

            # Tab 3: System Info
            with gr.TabItem("🖥️ 系统信息"):
                gr.Markdown("查看后端状态和异构调度报告。")
                refresh_btn = gr.Button("🔄 刷新", variant="secondary")
                backend_md = gr.Markdown()
                schedule_md = gr.Markdown()
                refresh_btn.click(
                    fn=get_system_info,
                    inputs=[simulate_npu_toggle],
                    outputs=[backend_md, schedule_md],
                )

            # Tab 4: Benchmark
            with gr.TabItem("⚡ 基准测试"):
                gr.Markdown(
                    "运行快速基准测试，查看各任务的后端分配和执行耗时。\n\n"
                    "> ⚠️ 所有 simulated backend 结果仅用于验证调度流程，"
                    "不可作为硬件性能结果。"
                )
                bench_btn = gr.Button("▶️ 运行基准测试", variant="primary")
                bench_output = gr.Markdown()
                bench_btn.click(
                    fn=run_benchmark,
                    inputs=[simulate_npu_toggle],
                    outputs=[bench_output],
                )

        gr.Markdown(
            """
---
*LocalDoc Agent v0.1 | 异构计算课程实验 | 当前为 CPU fallback + simulated backend 模式*
*所有 simulated backend 数据不代表真实 AMD 硬件性能*
            """
        )

    return app


def main():
    """Launch the Gradio app."""
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
