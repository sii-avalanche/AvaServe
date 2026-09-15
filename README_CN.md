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

**变更**

详见[我们的变更](log.md)
