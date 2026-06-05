#!/bin/bash
# Shared Python environment bootstrap for no-sudo Jupyter/Ubuntu containers.

bootstrap_python_env() {
    local python_cmd="$1"
    local venv_dir="$2"

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
