# Concurrency tuning guide:
#   --max-running-requests      max concurrency, adjust as needed
#   --cuda-graph-max-bs-decode  set to --max-running-requests / --pp-size + 2
NCCL_IB_HCA="=${HOST_RDMA_DEVICE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_OPT_FP8_WO_A_GEMM=1 \
SGLANG_PP_LAYER_PARTITION="16,16,16,13" \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve \
    --model-path ${DSV4_PRO_MODEL_PATH} \
    --served-model-name dsv4-pro \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --trust-remote-code \
    --tp-size 4 \
    --ep-size 4 \
    --pp-size 4 \
    --moe-runner-backend humming \
    --enforce-disable-flashinfer-allreduce-fusion \
    --speculative-algorithm DSPARK \
    --pp-prefill-delay-min-tokens 32768 \
    --pp-prefill-delay-max-passes 64 \
    --dist-init-addr ${SGLANG_DIST_ADDR} \
    --nnodes 2 \
    --node-rank ${SGLANG_DIST_RANK} \
    --host 0.0.0.0 \
    --port ${SGLANG_SERVER_PORT} \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 256 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 66 \
    --mem-fraction-static 0.90 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back
