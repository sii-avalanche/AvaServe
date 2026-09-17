# AvaServe

English | [简体中文](README_CN.md)

**High performance LLM serving deployment**

- Optimized for [Kimi K3](deployment/Kimi-K3), [DeepSeek V4 Pro](deployment/Kimi-K3), … models
- Optimized for LLM inference throughput on Hopper GPUs
- Optimized mixed precision (FP4/FP8) support on Hopper GPUs
- Optimized PP/TP/DP and HiCache performance

**Installation**

Follow these steps:

- CUDA 13:

  ```bash
  # Install setuptools
  pip install setuptools wheel_stub

  # Install sglang and its dependencies
  cd AvaServe/python
  pip install . --no-build-isolation

  # Install sglang router
  pip install maturin
  cd AvaServe/sgl-model-gateway/bindings/python
  maturin build --release --out dist --features vendored-openssl
  pip install dist/*.whl
  ```

- CUDA 12:

  ```bash
  # Install setuptools
  pip install setuptools wheel_stub

  # Install CUDA 12.9 dependencies
  pip install torch==2.13.0+cu129 torchaudio==2.11.0+cu129 torchvision==0.28.0+cu129 --index-url https://download.pytorch.org/whl/cu129
  pip install sglang-kernel==0.4.6.post1+cu129 sgl-deep-gemm==0.1.7+cu129 sgl-deep-ep==0.1.2+cu129 --index-url https://sgl-project.github.io/whl/cu129

  # Install sglang
  cd AvaServe/python
  cp pyproject_cu12.toml pyproject.toml
  pip install . --no-build-isolation

  # Install sglang router
  pip install maturin
  cd AvaServe/sgl-model-gateway/bindings/python
  maturin build --release --out dist --features vendored-openssl
  pip install dist/*.whl
  ```

**Changes**

See [our changes](log.md) for details
