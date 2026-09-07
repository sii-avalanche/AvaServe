NCCL_IB_HCA="=${HOST_RDMA_DEVICE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_PP_LAYER_PARTITION="48,45" \
sglang serve \
    --model-path ${KIMI_K3_MODEL_PATH} \
    --served-model-name kimi-k3 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --preferred-sampling-params '{"repetition_penalty": 1.05}' \
    --trust-remote-code \
    --tp-size 8 \
    --pp-size 2 \
    --ep-size 8 \
    --dcp-size 8 \
    --moe-runner-backend humming \
    --decode-attention-backend flashinfer \
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
    --cuda-graph-max-bs-decode 66 \
    --mem-fraction-static 0.89 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 4 \
    --hicache-write-policy write_back
