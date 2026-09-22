# AvaServe

[English](README.md) | 简体中文

**高性能 LLM 服务部署**

- 优化 [Kimi K3](deployment/Kimi-K3)、[DeepSeek V4 Pro](deployment/DSV4-Pro)、[GLM-5.3 FP8](deployment/GLM5-FP8) 等模型
- 优化 Hopper GPU 上的 LLM 推理吞吐
- 优化 Hopper GPU 上的混合精度（FP4/FP8）推理
- 优化 PP/TP/DP 与 HiCache 性能

**安装**

参照如下步骤：

- CUDA 13:

  ```bash
  # 安装 setuptools
  pip install setuptools wheel_stub

  # 安装 sglang 及依赖
  cd AvaServe/python
  pip install . --no-build-isolation
  pip install nvidia-nccl-cu13==2.30.7  # 升级 nccl 到 2.30.7
  pip install mooncake-transfer-engine

  # 安装 sglang router
  pip install maturin
  cd AvaServe/sgl-model-gateway/bindings/python
  maturin build --release --out dist --features vendored-openssl
  pip install dist/*.whl
  ```

- CUDA 12:

  ```bash
  # 安装 setuptools
  pip install setuptools wheel_stub

  # 安装CUDA 12.9依赖
  pip install torch==2.13.0+cu129 torchaudio==2.11.0+cu129 torchvision==0.28.0+cu129 --index-url https://download.pytorch.org/whl/cu129
  pip install sglang-kernel==0.4.6.post1+cu129 sgl-deep-gemm==0.1.7+cu129 sgl-deep-ep==0.1.2+cu129 --index-url https://sgl-project.github.io/whl/cu129

  # 安装 sglang 及依赖
  cd AvaServe/python
  cp pyproject_cu12.toml pyproject.toml
  pip install . --no-build-isolation
  pip install nvidia-nccl-cu12==2.30.7  # 升级 nccl 到 2.30.7
  pip install mooncake-transfer-engine

  # 安装 sglang router
  pip install maturin
  cd AvaServe/sgl-model-gateway/bindings/python
  maturin build --release --out dist --features vendored-openssl
  pip install dist/*.whl
  ```

**变更**

详见[我们的变更](log.md)
