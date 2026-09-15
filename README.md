# AvaServe

English | [简体中文](README_CN.md)

**High performance LLM serving deployment**

- Optimized for [Kimi K3](deployment/Kimi-K3), [DeepSeek V4 Pro](deployment/Kimi-K3), … models
- Optimized for LLM inference throughput on Hopper GPUs
- Optimized mixed precision (FP4/FP8) support on Hopper GPUs
- Optimized PP/TP/DP and HiCache performance

**Installation**

1. Install the official SGLang v0.5.19 (pin `sglang==0.5.19`) by following the [official installation guide](https://docs.sglang.io/docs/get-started/install).

2. Install AvaServe on top of it (all dependencies are already provided by step 1):

   CUDA 13:

   ```bash
   cd python
   pip install . --no-deps --no-build-isolation
   ```

   CUDA 12: swap in the cu12 pyproject first, then install the same way:

   ```bash
   cp python/pyproject_cu12.toml python/pyproject.toml
   cd python
   pip install . --no-deps --no-build-isolation
   ```

**Changes**

See [our changes](log.md) for details
