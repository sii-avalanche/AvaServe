ARG CUDA_VERSION=12.9.1
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1
ARG PIP_DEFAULT_INDEX
ARG PIP_TRUSTED_HOST
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/

ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# Python setup (build from source for exact version)
ARG PYTHON_VERSION=3.12.12
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt update && apt install -y --no-install-recommends \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
    && wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz \
    # Extract major.minor version (e.g., 3.12 from 3.12.12)
    && PYTHON_MM=$(echo ${PYTHON_VERSION} | cut -d. -f1,2) \
    # Remove any existing python symlinks and create new ones
    && rm -f /usr/bin/python3 /usr/bin/python /usr/bin/pip3 /usr/bin/pip \
    && ln -sf /usr/local/bin/python${PYTHON_MM} /usr/bin/python${PYTHON_MM} \
    && ln -sf /usr/local/bin/python${PYTHON_MM} /usr/bin/python3 \
    && ln -sf /usr/local/bin/python${PYTHON_MM} /usr/bin/python \
    && ln -sf /usr/local/bin/pip${PYTHON_MM} /usr/bin/pip3 \
    && ln -sf /usr/local/bin/pip${PYTHON_MM} /usr/bin/pip \
    # Verify installation
    && python3 --version \
    && pip3 --version

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # Core system utilities
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    # NCCL (needed for pynccl_allocator JIT compilation)
    libnccl2 \
    libnccl-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    # Development tools
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    pkg-config \
    bear \
    less \
    rdma-core \
    gnuplot \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Replace pip global index if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    pip3 config set global.index-url ${PIP_DEFAULT_INDEX}; \
    if echo "${PIP_DEFAULT_INDEX}" | grep -qE '^http://'; then \
        PIP_INDEX_HOST=$(echo "${PIP_DEFAULT_INDEX}" | sed -E 's#^https?://([^/]+)/?.*$#\1#'); \
        pip3 config set global.trusted-host "${PIP_INDEX_HOST}"; \
    fi; \
fi \
&& if [ -n "${PIP_TRUSTED_HOST}" ]; then \
    pip3 config set global.trusted-host "${PIP_TRUSTED_HOST}"; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install uv for Python package management
RUN pip3 install uv

# Install Rust (needed for sgl-model-gateway compilation)
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install NVIDIA development tools
RUN echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# diff-so-fancy
RUN curl --retry 3 --retry-delay 2 -LSso /usr/local/bin/diff-so-fancy \
        https://${GITHUB_ARTIFACTORY}/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /usr/local/bin/diff-so-fancy

# clang-format
RUN curl --retry 3 --retry-delay 2 -LSso /usr/local/bin/clang-format \
        https://${GITHUB_ARTIFACTORY}/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /usr/local/bin/clang-format

# clangd
RUN curl --retry 3 --retry-delay 2 -fsSL -o clangd.zip \
        https://${GITHUB_ARTIFACTORY}/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip \
    && unzip -q clangd.zip \
    && cp -r clangd_18.1.3/bin/* /usr/local/bin/ \
    && cp -r clangd_18.1.3/lib/* /usr/local/lib/ \
    && rm -rf clangd_18.1.3 clangd.zip

# CMake
RUN CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# Install just
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://just.systems/install.sh | \
    sed "s|https://github.com|https://${GITHUB_ARTIFACTORY}|g" | \
    bash -s -- --tag 1.42.4 --to /usr/local/bin

# Add yank script
COPY --chown=root:root --chmod=755 docker/configs/yank /usr/local/bin/yank

# Install oh-my-zsh and plugins
RUN sh -c "$(curl --retry 3 --retry-delay 2 -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Copy config files
COPY docker/configs/opt/.vimrc /opt/sglang/.vimrc
COPY docker/configs/opt/.tmux.conf /opt/sglang/.tmux.conf
COPY docker/configs/opt/.gitconfig /opt/sglang/.gitconfig
COPY docker/configs/.zshrc /root/.zshrc

# Create workspace directory
WORKDIR /sgl-workspace

CMD ["/bin/bash"]
