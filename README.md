# AvaServe

This is a fork of official [SGLang](https://github.com/sgl-project/sglang), with SII's optimizations, fixes, and reference deployment scripts.

## How to Install

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

## Reference Deployment Scripts

All deployment scripts are located in the [`deployment`](deployment/) directory. Models currently covered:

- Kimi-K3

## Optimizations and Fixes

The following changes are ported onto sglang v0.5.19 in the `sii-v0.5.19` branch:

- Support Anthropic /v1/messages gateway for regular and PD routing, with protocol and observability updates
- Support Responses API and Codex compatibility: named function tool choice, prior-response function-call replay, cancellation handling, default output-token limits
- Support latest Codex custom_tool_call / custom_tool_call_output items
- Support codex multi agent: flatten namespace-wrapped function tools and convert agent_message items
- Support no_think reasoning effort across request parsing, GLM45, Inkling, Mistral and Responses paths; restore reasoning close-token parsing
- Support SII docker documentation, simplified Dockerfile and uv install script
- Support cu12 dependencies: add python/pyproject_cu12.toml next to the untouched upstream pyproject.toml instead of patching it in place, versions adapted to v0.5.19 pins: flashinfer 0.6.18, humming-kernels 0.1.12, cutlass-dsl 4.6.2
- Support bf16 recurrent-state pools in the fused KDA decode kernel: TState template widening bf16 to fp32 on load and rounding RTNE on store
- Support zero copy with the flashinfer mxfp4 moe backend: honor a published moe output spec before allocating the combine output
- Support KIMI-K3 to use the humming MoE backend: situ activation in the humming runner and the mxfp4 weight-prep dispatch
- Support DCP for Kimi-K3 on Hopper via the flashinfer_mla backend: prefill follows decode onto FlashInfer-MLA so its updater swaps in the gathered dcp_kv_buffer via dcp_kv_indptr/indices; rejects speculative decoding on this path
- Fix prefill-delay slot accounting to use current waiting demand
- Fix Mooncake chunk-ready sends with exception protection
- Fix request resource release ordering: release before scheduler batch filtering; the already-released guard uses the post-sgl-project/sglang#29428 req.kv.holds_kv / is_kv_released state in place of the removed kv_committed_freed / kv_overallocated_freed fields
- Fix HiCache logical anchors: reject LogicalHostPool anchor without sidecar pool transfers
- Fix Responses streaming for Codex tool outputs and max effort streams: normalize function_call_output list-shaped outputs in Chat and Harmony paths, serialize full-response snapshots with SDK-accepted effort then restore the original on the wire
- Fix kimi k3 image input: leave max_tokens unset so the scheduler clamps after multimodal expansion; skip the multimodal prompt path for kimi_k3 chat encoding
- Fix k3 draft model cuda graph support and support skipping padding token with marlin backend: dp-local draft buckets skip attn-tp alignment via the shared forward_is_dp_local_draft helper, require_mlp_sync exempts the dp-local draft, and capture wires the replay-updated real per-rank counts into set_dp_buffer_len
- Fix output of kimi-k3 with hicache and dp: MAX_LEN-aware dp local info and ReplaySSM ring cursor reset on mamba load_back
- Fix pp with hicache: dedicated gloo hicache_sync_group spanning the (pp, attn_tp, attn_cp) ranks sharing an attention DP rank, async MIN ready-count reductions reaped pp_size rounds later with cumulative counts; adapted to main's ack_prefetch_queue, cache-linker path and buffer_pipeline flush
- Fix prefill cuda graph capture: filter buckets that violate the attn-tp divisibility of the MAX_LEN DP gather; fix hicache group creation to reuse the attn-tp cpu group or negotiate via create_custom_parallel_group
- Fix dsv4-pro dp16 garbaged output: a replicated TP1 shared expert never passes through the combine sum; compute it on local hidden unconditionally
- Fix the Rust extension build pulling the wrong torch wheel: drop the torch==2.13.0 pin from build-system.requires so the isolated build resolves torch from the deployment environment (cu12)
- Fix PP recv kernels stalling cooperative flashinfer MLA decode kernels: gate the output/proxy tensor recv on launch_event so the NCCL recv kernel only becomes device-resident after the just-launched forward completes, removing the ~50% decode-step inflation
- Fix garbled DP-shared-expert decode under CUDA graphs: propagate skip_shared_experts through DeepseekV2MoE.forward_normal_dual_stream and the flashinfer piecewise-graph wrapper so the replicated TP1 shared expert no longer runs twice in-graph
- Fix request leak on generate_request teardown: mark ReqState.dispatched in _send_one_request / _send_batch_request and abort every still-tracked dispatched rid before discarding its state
- Fix bench_serving UnboundLocalError: initialize stream_error before the streaming loop in async_request_sglang_generate, matching the other request paths
- Optimize MoE gemm: eliminate the reduce_scatter for replicated inputs
- Optimize K3 latent projections: shard down/up_proj over the attention-TP group on Hopper with one node-local tail reduction; one-collective DP assembly of MoE inputs
- Optimize pp overhead: the last rank consumes its outputs from a local stash instead of receiving them around the ring, small tensor-dict sends skip the allgather slicing below 1MiB, cudaGraphUpload after instantiate/update, fused cast emitting the dense buffer directly
- Optimize pp cpu overhead: receive and fan out requests while the just-launched batch runs on the GPU
- Revert upstream sgl-project/sglang#34284 (RecentPrefillBatchSizeTracker) to fix prefill delay, restoring the decaying max_prefill_bs high-watermark
