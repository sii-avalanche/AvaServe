# Guide for Separate Docker Image with Python venv

> This guide splits the Docker image into two parts:
> 1. **Base image** (`simplified.Dockerfile`): OS, CUDA toolchain, system libs, dev tools
> 2. **Python environment** (`uv_install.sh`): SGLang, DeepEP, Mooncake, Python packages in a venv
>
> This separation allows rebuilding the Python environment without rebuilding the base image.

## 1. Install Python dependencies with uv (on CPU cluster)

> This step should run on a CPU cluster with network access.

### Prerequisites

1. Install Python 3.12

```bash
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.12 python3.12-venv python3.12-dev
```

2. Install CUDA Toolkit

- CUDA 12.8.1
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --toolkit --silent --installpath=$HOME/cuda-12.8.1
```

- CUDA 12.9.1
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sh cuda_12.9.1_575.57.08_linux.run --toolkit --silent --installpath=$HOME/cuda-12.9.1
```

### Build venv

- CUDA 12.8.1
```bash
chmod +x uv_install.sh
export CUDA_HOME=$HOME/cuda-12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VERSION=12.8.1
export WORKSPACE=/sgl-workspace
export BUILD_TYPE=all
export SYSTEM_PYTHON=/usr/bin/python3.12
export PIP_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_STRATEGY=first-index
export UV_TORCH_BACKEND=cu128
rm -rf /sgl-workspace
./uv_install.sh --venv /path/to/env --name sglang-py312
```

- CUDA 12.9.1
```bash
chmod +x uv_install.sh
export CUDA_HOME=$HOME/cuda-12.9.1
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VERSION=12.9.1
export WORKSPACE=/sgl-workspace
export BUILD_TYPE=all
export SYSTEM_PYTHON=/usr/bin/python3.12
export PIP_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_STRATEGY=first-index
export UV_TORCH_BACKEND=cu129
export CUDA_PYTHON_VERSION=12.9.0
export SGLANG_REPO_PATH=/inspire/hdd/project/aisystem-and-infra/26008/sii-sglang
rm -rf /sgl-workspace
./uv_install.sh --venv /inspire/qb-ilm/project/aisystem-and-infra/public/env --name sglang-py312-ruonan-v0511
```

### DeepEP variants

For Grace Blackwell (GB200):
```bash
export GRACE_BLACKWELL=1
```

For Hopper SBO:
```bash
export HOPPER_SBO=1
```

### Optional: Install from local repo

```bash
export SGLANG_REPO_PATH=/path/to/local/sglang
./uv_install.sh --venv /path/to/env --name sglang-dev
```

### Optional: Install specific version

```bash
export SGL_VERSION=0.5.11
./uv_install.sh --venv /path/to/env --name sglang-v0.5.11
```

## 2. Build base Docker image

> This step should run on bare metal with network access.

```bash
# From the sglang repo root:
sudo nerdctl build --progress=plain \
    -f docker/sii-docker/simplified.Dockerfile \
    --build-arg CUDA_VERSION=12.9.1 \
    --build-arg PIP_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    --build-arg PYTHON_VERSION=3.12.12 \
    -t sglang:cu129-base-py312-v0.5.11 .

# If using a private Nexus mirror, replace PIP_DEFAULT_INDEX with the real
# hostname that is resolvable from inside the Docker build environment. For
# HTTP mirrors, also pass PIP_TRUSTED_HOST=<mirror-host>.

# Tag and push
sudo nerdctl tag sglang:cu129-base-py312-v0.5.11 docker-qb.sii.edu.cn/inspire-studio/syslab-sglang:cu129-base-py312-v0.5.11
sudo nerdctl login -u <harbor-username> docker-qb.sii.edu.cn
# For Harbor robot accounts, quote the username so the shell does not expand `$`:
# sudo nerdctl login -u 'robot$inspire-studio+user-...' docker-qb.sii.edu.cn
sudo nerdctl push docker-qb.sii.edu.cn/inspire-studio/syslab-sglang:cu129-base-py312-v0.5.11
```

## 3. Use the venv with Docker

```bash
source /path/to/env/sglang-py312/bin/activate
python -c "import sglang; print(sglang.__version__)"
deactivate
```

## Changes from previous version

Key updates in this version (aligned with main Dockerfile):

| Item | Old | New |
|---|---|---|
| sgl-kernel package name | `sgl-kernel` | `sglang-kernel` |
| sgl-kernel version | 0.3.21 | 0.4.0 |
| FlashInfer version | 0.6.3 | 0.6.6 |
| Gateway binary name | `sgl-model-gateway` | `sglang-router` |
| NCCL dev libs | not included | `libnccl2` + `libnccl-dev` |
| CUTLASS DSL | not included | `nvidia-cutlass-dsl` + `nvidia-cutlass-dsl-libs-base` |
| cuda-python | not included | `cuda-python` (12.9 or 13.2.0) |
| DeepEP variants | default only | + Grace Blackwell, + Hopper SBO |
| Triton ptxas fix | not included | Symlink for Blackwell sm_103a (CUDA 13) |
| flashinfer-jit-cache | not included | Optional via `INSTALL_FLASHINFER_JIT_CACHE=1` |
| SGL_VERSION | not supported | Clone specific version tag |
