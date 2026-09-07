#!/bin/bash
# uv_install.sh - Install SGLang and dependencies using uv
#
# Usage:
#   ./uv_install.sh --venv /path/to/venvs --name sglang
#
# This script:
#   1. Creates a virtual environment using system Python
#   2. Installs uv in the virtual environment
#   3. Uses uv to install all packages
#
# Environment variables:
#   CUDA_VERSION          - CUDA version (default: 12.8.1)
#   BUILD_TYPE            - SGLang build type: all, srt, openai (default: all)
#   SGL_KERNEL_VERSION    - sglang-kernel version (default: 0.4.3)
#   SGL_DEEP_GEMM_VERSION - sgl-deep-gemm version (default: 0.1.2)
#   FLASHINFER_VERSION    - FlashInfer version (default: 0.6.8.post1)
#   TORCH_VERSION         - PyTorch version (default: 2.11.0)
#   TORCHAUDIO_VERSION    - torchaudio version (default: 2.11.0)
#   TORCHVISION_VERSION   - Optional torchvision version pin
#   CUDA_PYTHON_VERSION   - cuda-python version (default: 12.9.0 for CUDA 12, 13.2.0 for CUDA 13)
#   WORKSPACE             - Workspace directory (default: /sgl-workspace)
#   SGLANG_REPO_PATH      - Optional local SGLang repo path to install from
#   SGLANG_REPO_URL       - SGLang git URL when cloning (default: upstream GitHub)
#   SGLANG_GIT_REF        - Optional git ref/commit/tag to checkout after clone
#   SGL_VERSION           - Specific version tag to clone (e.g., 0.5.11)
#   USE_LATEST_SGLANG     - Clone HEAD of main branch: 0 or 1 (default: 0)
#   SYSTEM_PYTHON         - System Python path (default: /usr/bin/python3.12)
#   SKIP_DEEPEP           - Skip DeepEP: 0 or 1 (default: 0)
#   SKIP_MOONCAKE         - Skip Mooncake: 0 or 1 (default: 0)
#   SKIP_GATEWAY          - Skip sgl-model-gateway: 0 or 1 (default: 0)
#   INSTALL_FLASHINFER_JIT_CACHE - Install flashinfer-jit-cache: 0 or 1 (default: 0)
#   GRACE_BLACKWELL       - Use Grace Blackwell DeepEP branch: 0 or 1 (default: 0)
#   HOPPER_SBO            - Use Hopper SBO DeepEP branch: 0 or 1 (default: 0)
#   PIP_DEFAULT_INDEX     - Override pip index-url for bootstrap installs
#   PIP_TRUSTED_HOST      - Optional trusted host for pip bootstrap installs
#   UV_DEFAULT_INDEX      - Override uv default index
#   UV_EXTRA_INDEX_URL    - Optional extra index for uv
#   UV_TORCH_BACKEND      - uv PyTorch backend (default: cu<CUDA index>)

set -e

# ============================================
# Parse command line arguments
# ============================================
VENV_DIR=""
VENV_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --name)
            VENV_NAME="$2"
            shift 2
            ;;
        --python)
            SYSTEM_PYTHON="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --venv DIR --name NAME [options]"
            echo ""
            echo "Options:"
            echo "  --venv DIR     Parent directory for virtual environment"
            echo "  --name NAME    Name of the virtual environment"
            echo "  --python PATH  System Python path (default: /usr/bin/python3.12)"
            echo "  --help         Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --venv /opt/venvs --name sglang"
            echo "  # Creates: /opt/venvs/sglang"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$VENV_DIR" ] || [ -z "$VENV_NAME" ]; then
    echo "ERROR: --venv and --name are required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# ============================================
# Default values
# ============================================
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
BUILD_TYPE="${BUILD_TYPE:-all}"
SGL_KERNEL_VERSION="${SGL_KERNEL_VERSION:-0.4.3}"
SGL_DEEP_GEMM_VERSION="${SGL_DEEP_GEMM_VERSION:-0.1.2}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.8.post1}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
CUDA_PYTHON_VERSION="${CUDA_PYTHON_VERSION:-}"
WORKSPACE="${WORKSPACE:-/sgl-workspace}"
SGLANG_REPO_PATH="${SGLANG_REPO_PATH:-}"
SGLANG_REPO_URL="${SGLANG_REPO_URL:-https://github.com/sgl-project/sglang.git}"
SGLANG_GIT_REF="${SGLANG_GIT_REF:-}"
SGL_VERSION="${SGL_VERSION:-}"
USE_LATEST_SGLANG="${USE_LATEST_SGLANG:-0}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3.12}"
SKIP_DEEPEP="${SKIP_DEEPEP:-0}"
SKIP_MOONCAKE="${SKIP_MOONCAKE:-0}"
SKIP_GATEWAY="${SKIP_GATEWAY:-0}"
INSTALL_FLASHINFER_JIT_CACHE="${INSTALL_FLASHINFER_JIT_CACHE:-0}"
BUILD_AND_DOWNLOAD_PARALLEL="${BUILD_AND_DOWNLOAD_PARALLEL:-8}"
GITHUB_ARTIFACTORY="${GITHUB_ARTIFACTORY:-github.com}"

# DeepEP variants
GRACE_BLACKWELL="${GRACE_BLACKWELL:-0}"
HOPPER_SBO="${HOPPER_SBO:-0}"
GRACE_BLACKWELL_DEEPEP_BRANCH="${GRACE_BLACKWELL_DEEPEP_BRANCH:-gb200_blog_part_2}"
HOPPER_SBO_DEEPEP_COMMIT="${HOPPER_SBO_DEEPEP_COMMIT:-9f2fc4b3182a51044ae7ecb6610f7c9c3258c4d6}"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee}"

MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.9}"
MOONCAKE_COMPILE_ARG="${MOONCAKE_COMPILE_ARG:--DUSE_HTTP=ON -DUSE_MNNVL=ON -DUSE_CUDA=ON -DWITH_EP=ON}"
UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-first-index}"
PIP_DEFAULT_INDEX="${PIP_DEFAULT_INDEX:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-${PIP_DEFAULT_INDEX}}"
UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-${PIP_EXTRA_INDEX_URL}}"

VENV_PATH="${VENV_DIR}/${VENV_NAME}"

# Determine CUDA index
case "$CUDA_VERSION" in
    12.6.1) CUINDEX=126 ;;
    12.8.1) CUINDEX=128 ;;
    12.9.1) CUINDEX=129 ;;
    13.0.1) CUINDEX=130 ;;
    *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;;
esac

CUDA_MAJOR="${CUDA_VERSION%%.*}"
CU_TAG="cu${CUINDEX}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-${CU_TAG}}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${CU_TAG}}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/${CU_TAG}/torch2.6}"

if [ -z "${CUDA_PYTHON_VERSION}" ]; then
    if [ "$CUDA_MAJOR" = "12" ]; then
        CUDA_PYTHON_VERSION="12.9.0"
    elif [ "$CUDA_MAJOR" = "13" ]; then
        CUDA_PYTHON_VERSION="13.2.0"
    fi
fi

case "$CUDA_VERSION" in
    12.6.1) SGL_KERNEL_CU_TAG=cu124 ;;
    12.8.1|12.9.1) SGL_KERNEL_CU_TAG=cu129 ;;
    13.0.1) SGL_KERNEL_CU_TAG=cu130 ;;
esac

echo "=============================================="
echo "SGLang Installation"
echo "=============================================="
echo "VENV_PATH: ${VENV_PATH}"
echo "SYSTEM_PYTHON: ${SYSTEM_PYTHON}"
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "CUDA_PYTHON_VERSION: ${CUDA_PYTHON_VERSION}"
echo "UV_TORCH_BACKEND: ${UV_TORCH_BACKEND}"
echo "PYTORCH_INDEX_URL: ${PYTORCH_INDEX_URL}"
echo "WORKSPACE: ${WORKSPACE}"
echo "SGLANG_REPO_PATH: ${SGLANG_REPO_PATH:-<auto-clone>}"
echo "SGLANG_REPO_URL: ${SGLANG_REPO_URL}"
echo "SGLANG_GIT_REF: ${SGLANG_GIT_REF:-<default-branch HEAD>}"
echo "SGL_VERSION: ${SGL_VERSION:-<not set>}"
echo "SGL_KERNEL_VERSION: ${SGL_KERNEL_VERSION}"
echo "SGL_DEEP_GEMM_VERSION: ${SGL_DEEP_GEMM_VERSION}"
echo "UV_INDEX_STRATEGY: ${UV_INDEX_STRATEGY}"
echo "PIP_DEFAULT_INDEX: ${PIP_DEFAULT_INDEX:-<system-default>}"
echo "UV_DEFAULT_INDEX: ${UV_DEFAULT_INDEX:-<system-default>}"
echo "GRACE_BLACKWELL: ${GRACE_BLACKWELL}"
echo "HOPPER_SBO: ${HOPPER_SBO}"
echo "=============================================="

# ============================================
# Step 0: Create virtual environment
# ============================================
echo ""
echo "=== [0/9] Creating virtual environment ==="

# Verify system Python exists
if [ ! -x "$SYSTEM_PYTHON" ]; then
    echo "ERROR: System Python not found at ${SYSTEM_PYTHON}"
    exit 1
fi

echo "System Python: ${SYSTEM_PYTHON}"
echo "Version: $(${SYSTEM_PYTHON} --version)"

mkdir -p "${VENV_DIR}"

# Create virtual environment if not exists
if [ -d "${VENV_PATH}" ]; then
    echo "Virtual environment already exists at ${VENV_PATH}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_PATH}"
        "${SYSTEM_PYTHON}" -m venv "${VENV_PATH}" || \
        "${SYSTEM_PYTHON}" -m virtualenv "${VENV_PATH}"
    fi
else
    "${SYSTEM_PYTHON}" -m venv "${VENV_PATH}" || \
    "${SYSTEM_PYTHON}" -m virtualenv "${VENV_PATH}"
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"
hash -r

if [ -n "${UV_DEFAULT_INDEX}" ]; then
    export UV_DEFAULT_INDEX
fi
if [ -n "${UV_EXTRA_INDEX_URL}" ]; then
    export UV_EXTRA_INDEX_URL
fi
export UV_TORCH_BACKEND

echo "Activated: ${VENV_PATH}"
echo "Python: $(which python)"
echo "Version: $(python --version)"

# Install pip and uv in venv
echo ""
echo "=== Installing pip and uv in venv ==="
PIP_BOOTSTRAP_ARGS=()

if [ -n "${PIP_DEFAULT_INDEX}" ]; then
    PIP_BOOTSTRAP_ARGS+=(--index-url "${PIP_DEFAULT_INDEX}")
fi

if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then
    PIP_BOOTSTRAP_ARGS+=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
fi

if [ -n "${PIP_TRUSTED_HOST}" ]; then
    PIP_BOOTSTRAP_ARGS+=(--trusted-host "${PIP_TRUSTED_HOST}")
fi

python -m pip install --upgrade pip "${PIP_BOOTSTRAP_ARGS[@]}"
pip install uv "${PIP_BOOTSTRAP_ARGS[@]}"

echo "uv version: $(uv --version)"

torch_cuda_tag_matches() {
    python - "$CU_TAG" <<'PY'
import sys

expected = sys.argv[1]
try:
    import torch
except Exception as exc:
    print(f"torch import failed before CUDA tag check: {exc}")
    raise SystemExit(1)

cuda = torch.version.cuda
if not cuda:
    print(f"torch CUDA is not available in metadata: torch={torch.__version__}, cuda={cuda}")
    raise SystemExit(1)

parts = cuda.split(".")
actual = f"cu{parts[0]}{parts[1]}"
if actual != expected:
    print(
        f"torch CUDA tag mismatch: torch={torch.__version__}, "
        f"torch.version.cuda={cuda}, expected={expected}"
    )
    raise SystemExit(1)

print(f"torch CUDA tag OK: torch={torch.__version__}, torch.version.cuda={cuda}")
PY
}

install_matching_pytorch() {
    local packages=("torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}")
    if [ -n "${TORCHVISION_VERSION}" ]; then
        packages+=("torchvision==${TORCHVISION_VERSION}")
    else
        packages+=("torchvision")
    fi

    local args=(
        "${packages[@]}"
        --index-url "${PYTORCH_INDEX_URL}"
        --index-strategy first-index
        --torch-backend "${UV_TORCH_BACKEND}"
        --reinstall
    )

    local fallback_index="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"
    if [ -n "${fallback_index}" ] && [ "${fallback_index}" != "${PYTORCH_INDEX_URL}" ]; then
        args+=(--extra-index-url "${fallback_index}")
    fi
    if [ -n "${UV_EXTRA_INDEX_URL}" ] \
        && [ "${UV_EXTRA_INDEX_URL}" != "${PYTORCH_INDEX_URL}" ] \
        && [ "${UV_EXTRA_INDEX_URL}" != "${fallback_index}" ]; then
        args+=(--extra-index-url "${UV_EXTRA_INDEX_URL}")
    fi

    echo "Installing PyTorch packages for ${CU_TAG} from ${PYTORCH_INDEX_URL}..."
    uv pip install "${args[@]}"
    torch_cuda_tag_matches
}

ensure_matching_pytorch() {
    if ! torch_cuda_tag_matches; then
        echo "Reinstalling PyTorch to match CUDA_VERSION=${CUDA_VERSION} (${CU_TAG})..."
        install_matching_pytorch
    fi
}

install_matching_sglang_kernel() {
    local kernel_version="$1"
    local wheel_url="https://${GITHUB_ARTIFACTORY}/sgl-project/whl/releases/download/v${kernel_version}/sglang_kernel-${kernel_version}+${SGL_KERNEL_CU_TAG}-cp310-abi3-manylinux2014_$(uname -m).whl"

    echo "Installing sglang-kernel ${kernel_version} for ${SGL_KERNEL_CU_TAG}..."
    uv pip install "${wheel_url}" --reinstall --no-deps
}

install_matching_sgl_deep_gemm() {
    local deep_gemm_version="$1"
    local wheel_url="https://${GITHUB_ARTIFACTORY}/sgl-project/whl/releases/download/v${deep_gemm_version}/sgl_deep_gemm-${deep_gemm_version}+${SGL_KERNEL_CU_TAG}-py3-none-manylinux2014_$(uname -m).whl"

    echo "Installing sgl-deep-gemm ${deep_gemm_version} for ${SGL_KERNEL_CU_TAG}..."
    uv pip install "${wheel_url}" --reinstall
}

read_sglang_kernel_version() {
    python - "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)

for dep in data.get("project", {}).get("dependencies", []):
    if dep.startswith("sglang-kernel=="):
        print(dep.split("==", 1)[1].split(";", 1)[0].strip())
        break
PY
}

read_sgl_deep_gemm_version() {
    python - "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)

for dep in data.get("project", {}).get("dependencies", []):
    if dep.startswith("sgl-deep-gemm=="):
        print(dep.split("==", 1)[1].split(";", 1)[0].strip())
        break
PY
}

prepare_uv_overrides() {
    UV_OVERRIDES_FILE=""

    if [ "$CUDA_MAJOR" = "12" ]; then
        UV_OVERRIDES_FILE="${WORKSPACE}/uv-overrides-${CU_TAG}.txt"
        printf "cuda-python==%s\n" "${CUDA_PYTHON_VERSION}" > "${UV_OVERRIDES_FILE}"
        echo "Using uv overrides: ${UV_OVERRIDES_FILE}"
    fi
}

# ============================================
# Step 1: Install base packages
# ============================================
echo ""
echo "=== [1/9] Installing base packages ==="
uv pip install pip setuptools setuptools-scm wheel html5lib six idna

echo ""
echo "=== [1.5/9] Installing PyTorch for ${CU_TAG} ==="
install_matching_pytorch

# ============================================
# Step 2: Install sglang-kernel
# ============================================
echo ""
echo "=== [2/9] Installing sglang-kernel and sgl-deep-gemm ==="
install_matching_sglang_kernel "${SGL_KERNEL_VERSION}"
install_matching_sgl_deep_gemm "${SGL_DEEP_GEMM_VERSION}"

# ============================================
# Step 3: Clone and install SGLang
# ============================================
echo ""
echo "=== [3/9] Installing SGLang ==="

mkdir -p "${WORKSPACE}"
prepare_uv_overrides

if [ -n "${SGLANG_REPO_PATH}" ]; then
    if [ ! -d "${SGLANG_REPO_PATH}" ]; then
        echo "ERROR: SGLANG_REPO_PATH does not exist: ${SGLANG_REPO_PATH}"
        exit 1
    fi
    if [ ! -f "${SGLANG_REPO_PATH}/python/pyproject.toml" ]; then
        echo "ERROR: SGLANG_REPO_PATH does not look like an sglang repo: ${SGLANG_REPO_PATH}"
        echo "Expected file: ${SGLANG_REPO_PATH}/python/pyproject.toml"
        exit 1
    fi
    SGLANG_SRC_DIR="${SGLANG_REPO_PATH}"
    echo "Using local SGLang repo: ${SGLANG_SRC_DIR}"
else
    SGLANG_SRC_DIR="${WORKSPACE}/sglang"
    if [ ! -d "${SGLANG_SRC_DIR}" ]; then
        if [ "$USE_LATEST_SGLANG" = "1" ]; then
            echo "Cloning SGLang HEAD from ${SGLANG_REPO_URL}..."
            git clone --depth=1 "${SGLANG_REPO_URL}" "${SGLANG_SRC_DIR}"
        elif [ -n "${SGL_VERSION}" ]; then
            echo "Cloning SGLang v${SGL_VERSION} from ${SGLANG_REPO_URL}..."
            git clone --depth=1 --branch "v${SGL_VERSION}" "${SGLANG_REPO_URL}" "${SGLANG_SRC_DIR}"
        else
            echo "Cloning SGLang from ${SGLANG_REPO_URL}..."
            git clone --depth=1 "${SGLANG_REPO_URL}" "${SGLANG_SRC_DIR}"
        fi
        if [ -n "${SGLANG_GIT_REF}" ]; then
            git -C "${SGLANG_SRC_DIR}" fetch --depth=1 origin "${SGLANG_GIT_REF}"
            git -C "${SGLANG_SRC_DIR}" checkout FETCH_HEAD
        fi
    else
        echo "SGLang directory already exists at ${SGLANG_SRC_DIR}, skipping clone"
        if [ -n "${SGLANG_GIT_REF}" ]; then
            echo "WARNING: SGLANG_GIT_REF is ignored because ${SGLANG_SRC_DIR} already exists"
        fi
    fi
fi

cd "${SGLANG_SRC_DIR}"
# Install sglang (non-editable mode for portability)
INSTALL_ARGS=("./python[${BUILD_TYPE}]")

if [ -n "${PYTORCH_INDEX_URL}" ]; then
    INSTALL_ARGS+=(--extra-index-url "${PYTORCH_INDEX_URL}")
fi

if [ -n "${FLASHINFER_INDEX_URL}" ]; then
    INSTALL_ARGS+=(--extra-index-url "${FLASHINFER_INDEX_URL}")
fi

INSTALL_ARGS+=(--index-strategy "${UV_INDEX_STRATEGY}")
INSTALL_ARGS+=(--torch-backend "${UV_TORCH_BACKEND}")
if [ -n "${UV_OVERRIDES_FILE}" ]; then
    INSTALL_ARGS+=(--overrides "${UV_OVERRIDES_FILE}")
fi
INSTALL_ARGS+=(--no-build-isolation)

uv pip install "${INSTALL_ARGS[@]}"
ensure_matching_pytorch

SGL_KERNEL_VERSION_FROM_SGLANG="$(read_sglang_kernel_version "${SGLANG_SRC_DIR}/python/pyproject.toml")"
if [ -n "${SGL_KERNEL_VERSION_FROM_SGLANG}" ]; then
    install_matching_sglang_kernel "${SGL_KERNEL_VERSION_FROM_SGLANG}"
fi

SGL_DEEP_GEMM_VERSION_FROM_SGLANG="$(read_sgl_deep_gemm_version "${SGLANG_SRC_DIR}/python/pyproject.toml")"
if [ -n "${SGL_DEEP_GEMM_VERSION_FROM_SGLANG}" ]; then
    install_matching_sgl_deep_gemm "${SGL_DEEP_GEMM_VERSION_FROM_SGLANG}"
fi

# Download flashinfer cubin
echo "Downloading flashinfer cubin..."
FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} FLASHINFER_LOGGING_LEVEL=warning python -m flashinfer --download-cubin || true

# Install flashinfer-jit-cache if requested
if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then
    echo "Installing flashinfer-jit-cache..."
    uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
        --extra-index-url "https://flashinfer.ai/whl/cu${CUINDEX}"
fi

# ============================================
# Step 4: Install DeepEP
# ============================================
if [ "$SKIP_DEEPEP" != "1" ]; then
    echo ""
    echo "=== [4/9] Installing DeepEP ==="

    cd "${WORKSPACE}"

    if [ ! -d "${WORKSPACE}/DeepEP" ]; then
        if [ "$GRACE_BLACKWELL" = "1" ]; then
            echo "Cloning Grace Blackwell DeepEP branch..."
            git clone "https://${GITHUB_ARTIFACTORY}/fzyzcjy/DeepEP.git" DeepEP
            cd DeepEP
            git checkout "${GRACE_BLACKWELL_DEEPEP_BRANCH}"
        elif [ "$HOPPER_SBO" = "1" ]; then
            echo "Cloning Hopper SBO DeepEP branch..."
            git clone "https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP.git" -b antgroup-opt DeepEP
            cd DeepEP
            git checkout "${HOPPER_SBO_DEEPEP_COMMIT}"
        else
            curl --retry 3 --retry-delay 2 -fsSL -o ${DEEPEP_COMMIT}.zip \
                "https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip"
            unzip -q ${DEEPEP_COMMIT}.zip && rm ${DEEPEP_COMMIT}.zip
            mv DeepEP-${DEEPEP_COMMIT} DeepEP
            cd DeepEP
        fi

        # Patch timeout values
        sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh
        sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh
    else
        echo "DeepEP directory already exists, skipping clone"
        cd "${WORKSPACE}/DeepEP"
    fi

    # Determine CUDA arch list
    case "$CUDA_VERSION" in
        12.6.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' ;;
        12.8.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0' ;;
        12.9.1|13.0.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' ;;
    esac

    # CUDA 13 specific patch
    if [ "$CUDA_MAJOR" = "13" ]; then
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py 2>/dev/null || true
    fi

    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} pip install --no-build-isolation .
else
    echo ""
    echo "=== [4/9] Skipping DeepEP (SKIP_DEEPEP=1) ==="
fi

# ============================================
# Step 5: Install Mooncake
# ============================================
if [ "$SKIP_MOONCAKE" != "1" ]; then
    echo ""
    echo "=== [5/9] Installing Mooncake ==="

    cd "${WORKSPACE}"

    if [ "$CUDA_MAJOR" -ge 13 ]; then
        echo "CUDA >= 13, building mooncake-transfer-engine from source..."
        if [ ! -d "${WORKSPACE}/Mooncake" ]; then
            git clone --branch "v${MOONCAKE_VERSION}" --depth 1 https://github.com/kvcache-ai/Mooncake.git
        fi
        cd Mooncake
        bash dependencies.sh
        mkdir -p build && cd build
        cmake .. ${MOONCAKE_COMPILE_ARG}
        make -j$(nproc)
        make install
    else
        echo "CUDA < 13, installing mooncake-transfer-engine via uv..."
        uv pip install "mooncake-transfer-engine==${MOONCAKE_VERSION}"
    fi
else
    echo ""
    echo "=== [5/9] Skipping Mooncake (SKIP_MOONCAKE=1) ==="
fi

# ============================================
# Step 6: Install sgl-model-gateway
# ============================================
GATEWAY_DIR="${SGLANG_SRC_DIR}/sgl-model-gateway"
if [ "$SKIP_GATEWAY" = "1" ]; then
    echo ""
    echo "=== [6/9] Skipping sgl-model-gateway (SKIP_GATEWAY=1) ==="
elif [ ! -d "${GATEWAY_DIR}" ]; then
    echo ""
    echo "=== [6/9] Skipping sgl-model-gateway (directory not found: ${GATEWAY_DIR}) ==="
else
    echo ""
    echo "=== [6/9] Building sgl-model-gateway ==="

    # Install protoc if not available
    if ! command -v protoc &> /dev/null; then
        echo "Installing protoc..."
        PROTOC_VERSION="${PROTOC_VERSION:-25.1}"
        ARCH=$(uname -m)
        if [ "$ARCH" = "x86_64" ]; then
            PROTOC_ARCH="linux-x86_64"
        elif [ "$ARCH" = "aarch64" ]; then
            PROTOC_ARCH="linux-aarch_64"
        fi

        curl --retry 3 --retry-delay 2 -fsSL -o /tmp/protoc.zip \
            "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-${PROTOC_ARCH}.zip"

        mkdir -p ~/.local/bin ~/.local/include
        unzip -o /tmp/protoc.zip -d ~/.local
        rm /tmp/protoc.zip
        export PATH="$HOME/.local/bin:$PATH"
        echo "protoc installed: $(protoc --version)"
    fi

    # Install Rust if not available
    if ! command -v cargo &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    uv pip install maturin

    cd "${GATEWAY_DIR}/bindings/python"
    ulimit -n 65536 2>/dev/null || true
    maturin build --release --features vendored-openssl --out dist
    pip install --force-reinstall dist/*.whl

    cd "${GATEWAY_DIR}"
    cargo build --release --bin sgl-model-gateway --features vendored-openssl

    # Install binary
    cp target/release/sgl-model-gateway "${VENV_PATH}/bin/sgl-model-gateway"

    # Cleanup
    rm -rf target dist
fi

# ============================================
# Step 7: Install additional packages
# ============================================
echo ""
echo "=== [7/9] Installing additional packages ==="

uv pip install \
    datamodel_code_generator \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    wheel \
    scikit-build-core \
    nixl \
    py-spy \
    cubloaty \
    google-cloud-storage \
    pandas \
    matplotlib \
    tabulate \
    termplotlib

# ============================================
# Step 8: Install CUTLASS DSL packages
# ============================================
echo ""
echo "=== [8/9] Installing CUTLASS DSL and patches ==="

uv pip install "nvidia-cutlass-dsl>=4.4.1" "nvidia-cutlass-dsl-libs-base>=4.4.1" --reinstall --no-deps

# NVIDIA packages patch
echo "Patching NVIDIA packages..."
if [ "$CUDA_MAJOR" = "12" ]; then
    uv pip install nvidia-nccl-cu12==2.28.3 --reinstall --no-deps
    uv pip install nvidia-cudnn-cu12==9.16.0.29 --reinstall --no-deps
    uv pip install "cuda-python==${CUDA_PYTHON_VERSION}"
elif [ "$CUDA_MAJOR" = "13" ]; then
    uv pip install nvidia-nccl-cu13==2.28.3 --reinstall --no-deps
    uv pip install nvidia-cudnn-cu13==9.16.0.29 --reinstall --no-deps
    uv pip install nvidia-cublas==13.1.0.3 --reinstall --no-deps
    uv pip install nixl-cu13 --no-deps
    uv pip install "cuda-python==${CUDA_PYTHON_VERSION}"
fi

# Fix Triton to use system ptxas for Blackwell (sm_103a) support (CUDA 13+ only)
if [ "$CUDA_MAJOR" = "13" ]; then
    TRITON_PTXAS_DIR=$(python -c "import triton; import os; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/bin'))" 2>/dev/null || true)
    if [ -n "$TRITON_PTXAS_DIR" ] && [ -d "$TRITON_PTXAS_DIR" ]; then
        echo "Fixing Triton ptxas symlink for Blackwell..."
        rm -f "${TRITON_PTXAS_DIR}/ptxas"
        ln -s /usr/local/cuda/bin/ptxas "${TRITON_PTXAS_DIR}/ptxas"
    fi
fi

# Upgrade urllib3
uv pip install --upgrade "urllib3>=2.6.3"

# ============================================
# Step 9: Verification
# ============================================
echo ""
echo "=== [9/9] Verifying installation ==="

python -c "import sglang; print(f'sglang version: {sglang.__version__}')" 2>/dev/null || echo "WARNING: sglang import failed"
python -c "import torch; print(f'torch version: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || echo "WARNING: torch import failed"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Virtual environment: ${VENV_PATH}"
echo "Workspace: ${WORKSPACE}"
echo ""
echo "To use on GPU node:"
echo "  1. Transfer ${VENV_PATH} and ${WORKSPACE} to GPU node"
echo "  2. source ${VENV_PATH}/bin/activate"
echo "  3. python -m sglang.launch_server ..."
echo "=============================================="
