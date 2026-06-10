from pathlib import Path

from experiments.resource_monitor import _parse_rocm_power_watts
from experiments.rocm_tools_profile import _build_rocprofv3_probe_command


def test_parse_rocm_smi_power_watts_colon_format():
    output = """
=================================== Power Consumption ====================================
GPU[0]          : Current Socket Graphics Package Power (W): 23.08
"""
    assert _parse_rocm_power_watts(output) == [23.08]


def test_parse_rocm_smi_power_watts_suffix_format():
    output = "GPU[0] : Average Graphics Package Power: 17.5 W"
    assert _parse_rocm_power_watts(output) == [17.5]


def test_rocprofv3_probe_command_uses_separator_before_application():
    command = _build_rocprofv3_probe_command(
        executable="/opt/rocm/bin/rocprofv3",
        python_exe="python",
        probe_script=Path("/tmp/probe.py"),
        profiler_dir=Path("/tmp/rocprof_out"),
    )
    separator_index = command.index("--")
    assert command[separator_index + 1 :] == ["python", "/tmp/probe.py"]
    assert "--runtime-trace" in command[:separator_index]
