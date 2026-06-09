#!/bin/bash
# Repair common ROCm runtime path issues in managed AMD/Jupyter containers.
#
# Some PyTorch ROCm wheels can see the AMD GPU but then crash on the first
# tensor operation because the ROCm runtime expects:
#   /opt/amdgpu/share/libdrm/amdgpu.ids
# This script tries to provide that file from the torch wheel or system libdrm.

set -euo pipefail

TARGET="/opt/amdgpu/share/libdrm/amdgpu.ids"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${GREEN}[info]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err() { echo -e "${RED}[error]${NC} $*"; }

find_python() {
    if command -v python >/dev/null 2>&1; then
        echo "python"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        return 1
    fi
}

find_amdgpu_ids() {
    local python_cmd="$1"
    local py_candidates

    py_candidates="$("$python_cmd" - <<'PY' 2>/dev/null || true
from pathlib import Path

candidates = []
try:
    import torch
    candidates.append(Path(torch.__file__).resolve().parent / "share" / "libdrm" / "amdgpu.ids")
except Exception:
    pass

for pattern in [
    ".venv/lib/python*/site-packages/torch/share/libdrm/amdgpu.ids",
    "/usr/local/lib/python*/dist-packages/torch/share/libdrm/amdgpu.ids",
    "/usr/local/lib/python*/site-packages/torch/share/libdrm/amdgpu.ids",
]:
    candidates.extend(Path("/").glob(pattern.lstrip("/")) if pattern.startswith("/") else Path(".").glob(pattern))

for path in candidates:
    if path.exists():
        print(path.resolve())
        break
PY
)"

    if [ -n "$py_candidates" ]; then
        echo "$py_candidates" | head -n 1
        return 0
    fi

    for path in \
        "/usr/share/libdrm/amdgpu.ids" \
        "/opt/rocm/share/libdrm/amdgpu.ids" \
        "/opt/amdgpu/share/libdrm/amdgpu.ids"
    do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

run_tensor_probe() {
    local python_cmd="$1"
    "$python_cmd" - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
y = (x * x).sum()
torch.cuda.synchronize()
print("ROCm tensor probe ok:", float(y.cpu()))
PY
}

main() {
    local python_cmd
    if ! python_cmd="$(find_python)"; then
        err "Python not found."
        return 1
    fi

    info "Python: $($python_cmd --version 2>&1)"
    info "Checking ROCm runtime helper file: $TARGET"

    if [ -f "$TARGET" ]; then
        info "Found $TARGET"
    else
        local src
        if ! src="$(find_amdgpu_ids "$python_cmd")"; then
            warn "Cannot find amdgpu.ids in the torch wheel or system libdrm paths."
            warn "Ask the platform admin to install/fix the ROCm package that provides $TARGET."
        else
            info "Found source amdgpu.ids: $src"
            if mkdir -p "$(dirname "$TARGET")" 2>/dev/null && ln -sf "$src" "$TARGET" 2>/dev/null; then
                info "Linked $TARGET -> $src"
            elif mkdir -p "$(dirname "$TARGET")" 2>/dev/null && cp "$src" "$TARGET" 2>/dev/null; then
                info "Copied $src -> $TARGET"
            else
                warn "No permission to write $TARGET"
                warn "Run this with container/admin permission, or ask the AMD/Jupyter platform owner to execute:"
                echo "  mkdir -p $(dirname "$TARGET")"
                echo "  ln -sf $src $TARGET"
            fi
        fi
    fi

    info "Running a minimal ROCm tensor probe in a child Python process ..."
    if run_tensor_probe "$python_cmd"; then
        info "ROCm tensor probe passed."
        return 0
    fi

    warn "ROCm tensor probe still failed."
    warn "This is a ROCm/container/runtime issue, not a LocalDoc Python-code issue."
    warn "If $TARGET exists and this still fails, the AMD driver/ROCm stack in the container must be fixed."
    return 0
}

main "$@"
