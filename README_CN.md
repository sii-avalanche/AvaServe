# AvaServe

[English](README.md) | 简体中文

**高性能 LLM 服务部署**

- 优化 [Kimi K3](deployment/Kimi-K3)、[DeepSeek V4 Pro](deployment/Kimi-K3) 等模型
- 优化 Hopper GPU 上的 LLM 推理吞吐
- 优化 Hopper GPU 上的混合精度（FP4/FP8）推理
- 优化 PP/TP/DP 与 HiCache 性能

**安装**

1. 按照[官方安装指南](https://docs.sglang.io/docs/get-started/install)安装官方 SGLang v0.5.19（固定版本 `sglang==0.5.19`）。

2. 在其基础上安装 AvaServe（所有依赖均已在第 1 步中提供）：

   CUDA 13：

   ```bash
   cd python
   pip install . --no-deps --no-build-isolation
   ```

   CUDA 12：先替换为 cu12 的 pyproject，然后按同样方式安装：

   ```bash
   cp python/pyproject_cu12.toml python/pyproject.toml
   cd python
   pip install . --no-deps --no-build-isolation
   ```

或者，如果官方的0.5.19的安装指南已被删除，参照如下步骤：

- CUDA 12:

  ```bash
  # 安装CUDA 12.9依赖
  pip install torch==2.13.0+cu129 torchaudio==2.11.0+cu129 torchvision==0.28.0+cu129 --index-url https://download.pytorch.org/whl/cu129
  pip install sglang-kernel==0.4.6.post1+cu129 sgl-deep-gemm==0.1.7+cu129 sgl-deep-ep==0.1.2+cu129 --index-url https://sgl-project.github.io/whl/cu129

  # 安装 sglang
  cd AvaServe/python
  cp pyproject_cu12.toml pyproject.toml
  pip install . --no-build-isolation

  # 安装 sglang router
  pip install maturin
  cd AvaServe/sgl-model-gateway/bindings/python
  maturin build --release --out dist --features vendored-openssl
  pip install dist/*.whl
  ```

**变更**

详见[我们的变更](log.md)
