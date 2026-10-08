# Concurrency tuning guide:
#   --max-running-requests      max concurrency, adjust as needed
#   --max-mamba-cache-size      set to 5x --max-running-requests
#   --cuda-graph-max-bs-decode  set to --max-running-requests
NCCL_IB_HCA="=${HOST_RDMA_DEVICE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
sglang serve \
    --model-path ${NEX_N25_PRO_MODEL_PATH} \
    --served-model-name nex-n2.5-pro \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chat-template ${NEX_N25_PRO_MODEL_PATH}/chat_template.jinja \
    --trust-remote-code \
    --tp-size 8 \
    --dcp-size 4 \
    --attention-backend fa3 \
    --enable-fused-moe-sum-all-reduce \
    --flashinfer-allreduce-fusion-backend mnnvl \
    --host 0.0.0.0 \
    --port ${SGLANG_SERVER_PORT} \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 256 \
    --max-mamba-cache-size 1280 \
    --cuda-graph-max-bs-decode 256 \
    --mem-fraction-static 0.89 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back
