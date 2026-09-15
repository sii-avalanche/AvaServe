# Concurrency tuning guide:
#   --max-running-requests      max concurrency, adjust as needed
#   --max-mamba-cache-size      set to 4x --max-running-requests
#   --cuda-graph-max-bs-decode  set to --max-running-requests / --pp-size + 2
NCCL_IB_HCA="=${HOST_RDMA_DEVICE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_PP_LAYER_PARTITION="48,45" \
sglang serve \
    --model-path ${KIMI_K3_MODEL_PATH} \
    --served-model-name kimi-k3 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --trust-remote-code \
    --tp-size 8 \
    --ep-size 8 \
    --pp-size 2 \
    --dcp-size 8 \
    --moe-runner-backend humming \
    --decode-attention-backend flashinfer \
    --pp-prefill-delay-min-tokens 32768 \
    --pp-prefill-delay-max-passes 32 \
    --dist-init-addr ${SGLANG_DIST_ADDR} \
    --nnodes 2 \
    --node-rank ${SGLANG_DIST_RANK} \
    --host 0.0.0.0 \
    --port ${SGLANG_SERVER_PORT} \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 128 \
    --max-mamba-cache-size 512 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 66 \
    --mem-fraction-static 0.89 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 4 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back
