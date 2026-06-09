#!/bin/bash
# Shared Python environment bootstrap for no-sudo Jupyter/Ubuntu containers.

_localdoc_info() {
    if declare -F info >/dev/null 2>&1; then
        info "$*"
    else
        echo "[信息] $*"
    fi
}

_localdoc_warn() {
    if declare -F warn >/dev/null 2>&1; then
        warn "$*"
    else
        echo "[警告] $*"
    fi
}

_localdoc_error() {
    if declare -F error >/dev/null 2>&1; then
        error "$*"
    else
        echo "[错误] $*"
    fi
}

_localdoc_truthy_env() {
    local value
    value="$(printf '%s' "${!1:-}" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

localdoc_current_python_rocm_probe_ok() {
    local python_cmd="$1"
    local project_root="${2:-$(pwd)}"
    PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" "$python_cmd" - <<'PY' >/tmp/localdoc_current_python_rocm_probe.log 2>&1
from localdoc.backends.rocm_safety import rocm_tensor_probe
import torch

ok, note = rocm_tensor_probe()
print("python:", __import__("sys").executable)
print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda_available:", torch.cuda.is_available())
print("probe_ok:", ok)
print("probe_note:", note)
raise SystemExit(0 if ok else 1)
PY
}

localdoc_prefer_current_python_for_rocm() {
    local python_cmd="$1"
    local project_root="${2:-$(pwd)}"
    if localdoc_current_python_rocm_probe_ok "$python_cmd" "$project_root"; then
        export LOCALDOC_USE_CURRENT_PYTHON=1
        _localdoc_info "当前 Python 的 ROCm tensor probe 通过，将跳过 .venv 以复用已验证的 ROCm 环境。"
        _localdoc_info "probe 详情: /tmp/localdoc_current_python_rocm_probe.log"
        return 0
    fi
    _localdoc_warn "当前 Python 的 ROCm tensor probe 未通过，将继续使用项目虚拟环境。"
    _localdoc_warn "probe 详情: /tmp/localdoc_current_python_rocm_probe.log"
    return 1
}

_localdoc_use_current_python_env() {
    local python_cmd="$1"

    if ! "$python_cmd" -m pip --version >/dev/null 2>&1; then
        _localdoc_error "当前 Python 没有 pip，且已要求跳过虚拟环境。"
        _localdoc_error "请在 Jupyter 终端先尝试: python3 -m pip --version"
        return 1
    fi

    LOCALDOC_PYTHON_CMD="$python_cmd"
    LOCALDOC_PIP_BREAK_FLAG=""
    if "$python_cmd" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
        LOCALDOC_PIP_BREAK_FLAG="--break-system-packages"
    fi
    export LOCALDOC_PYTHON_CMD LOCALDOC_PIP_BREAK_FLAG

    python() {
        "$LOCALDOC_PYTHON_CMD" "$@"
    }

    pip() {
        if [ "${1:-}" = "install" ]; then
            shift
            if [ -n "${LOCALDOC_PIP_BREAK_FLAG:-}" ]; then
                "$LOCALDOC_PYTHON_CMD" -m pip install --user "$LOCALDOC_PIP_BREAK_FLAG" "$@"
            else
                "$LOCALDOC_PYTHON_CMD" -m pip install --user "$@"
            fi
        else
            "$LOCALDOC_PYTHON_CMD" -m pip "$@"
        fi
    }

    _localdoc_info "使用当前 Python: $(python --version)"
}

bootstrap_python_env() {
    local python_cmd="$1"
    local venv_dir="$2"

    if _localdoc_truthy_env LOCALDOC_USE_CURRENT_PYTHON || _localdoc_truthy_env LOCALDOC_SKIP_VENV; then
        _localdoc_warn "LOCALDOC_USE_CURRENT_PYTHON/LOCALDOC_SKIP_VENV 已启用，跳过虚拟环境: $venv_dir"
        _localdoc_use_current_python_env "$python_cmd"
        return $?
    fi

    if [ -d "$venv_dir" ] && [ ! -f "$venv_dir/bin/activate" ]; then
        warn "检测到残缺虚拟环境: $venv_dir，删除后重新创建。"
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        info "创建虚拟环境: $venv_dir ..."
        if "$python_cmd" -m venv "$venv_dir"; then
            info "虚拟环境创建完成。"
        else
            warn "标准 venv 创建失败，尝试使用 virtualenv（不需要 sudo）。"
            rm -rf "$venv_dir"
            if "$python_cmd" -m pip --version >/dev/null 2>&1; then
                if "$python_cmd" -m pip install --user --quiet virtualenv; then
                    if "$python_cmd" -m virtualenv "$venv_dir"; then
                        info "virtualenv 创建完成。"
                    else
                        warn "virtualenv 创建失败，将使用当前 Python 用户环境。"
                    fi
                else
                    warn "安装 virtualenv 失败，将使用当前 Python 用户环境。"
                fi
            else
                warn "当前 Python 没有 pip，将使用当前 Python 环境并跳过自动依赖安装。"
            fi
        fi
    fi

    if [ -f "$venv_dir/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$venv_dir/bin/activate"
        info "已激活虚拟环境: $(python --version)"
        return 0
    fi

    warn "未能创建虚拟环境，切换为当前用户 Python 环境。"
    warn "依赖会安装到用户目录；Jupyter/no-sudo 容器可使用该模式。"

    if ! "$python_cmd" -m pip --version >/dev/null 2>&1; then
        error "当前 Python 没有 pip，且无法创建虚拟环境。"
        error "请在 Jupyter 终端先尝试: python3 -m pip --version"
        return 1
    fi

    LOCALDOC_PYTHON_CMD="$python_cmd"
    LOCALDOC_PIP_BREAK_FLAG=""
    if "$python_cmd" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
        LOCALDOC_PIP_BREAK_FLAG="--break-system-packages"
    fi
    export LOCALDOC_PYTHON_CMD LOCALDOC_PIP_BREAK_FLAG

    python() {
        "$LOCALDOC_PYTHON_CMD" "$@"
    }

    pip() {
        if [ "${1:-}" = "install" ]; then
            shift
            if [ -n "${LOCALDOC_PIP_BREAK_FLAG:-}" ]; then
                "$LOCALDOC_PYTHON_CMD" -m pip install --user "$LOCALDOC_PIP_BREAK_FLAG" "$@"
            else
                "$LOCALDOC_PYTHON_CMD" -m pip install --user "$@"
            fi
        else
            "$LOCALDOC_PYTHON_CMD" -m pip "$@"
        fi
    }

    info "使用当前 Python: $(python --version)"
}
