# Concurrency tuning guide:
#   --max-running-requests      max concurrency, adjust as needed
#   --cuda-graph-max-bs-decode  set to --max-running-requests / --pp-size + 2
# The GLM-5.3 DSPARK draft model is converted from https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark
NCCL_IB_HCA="=${HOST_RDMA_DEVICE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_PP_LAYER_PARTITION="21,21,21,15" \
sglang serve \
    --model-path ${GLM5_MODEL_PATH} \
    --served-model-name glm-5.3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --tp-size 2 --moe-dense-tp-size 1 \
    --ep-size 2 \
    --pp-size 4 \
    --moe-runner-backend deep_gemm \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path ${GLM5_DSPARK_MODEL_PATH} \
    --speculative-dspark-block-size 3 \
    --pp-prefill-delay-min-tokens 32768 \
    --pp-prefill-delay-max-passes 32 \
    --host 0.0.0.0 \
    --port ${SGLANG_SERVER_PORT} \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 10 \
    --mem-fraction-static 0.90 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 4 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back
