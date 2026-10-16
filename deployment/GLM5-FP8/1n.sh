# Concurrency tuning guide:
#   --max-running-requests      max concurrency, adjust as needed
#   --cuda-graph-max-bs-decode  set to --max-running-requests / --pp-size + 2
NCCL_IB_HCA="=${HOST_RDMA_DEVICE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_PP_LAYER_PARTITION="40,38" \
sglang serve \
    --model-path ${GLM5_MODEL_PATH} \
    --served-model-name glm-5.3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --tp-size 4 --moe-dense-tp-size 1 \
    --dcp-size 4 \
    --pp-size 2 \
    --dcp-replicate-q-proj \
    --dsa-prefill-backend fa3 \
    --dsa-decode-backend fa3 \
    --moe-runner-backend deep_gemm \
    --pp-prefill-delay-min-tokens 32768 \
    --pp-prefill-delay-max-passes 32 \
    --host 0.0.0.0 \
    --port ${SGLANG_SERVER_PORT} \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 128 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 66 \
    --mem-fraction-static 0.85 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 4 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back
