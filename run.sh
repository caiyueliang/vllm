if [ -z "$MODEL_PATH" ]; then
    export MODEL_PATH="/llm-data/0906"
fi
if [ -z "$DATA_TYPE" ]; then
    export DATA_TYPE="auto"
fi
if [ -z "$MAX_NUM_SEQS" ]; then
    export MAX_NUM_SEQS=256
fi
if [ -z "$BLOCK_SIZE" ]; then
    export BLOCK_SIZE=16
fi
if [ -z "$GPU_MEMORY_UTILIZATION" ]; then
    export GPU_MEMORY_UTILIZATION=0.98
fi
if [ -z "$PIPELINE_PARALLEL_SIZE" ]; then
    export PIPELINE_PARALLEL_SIZE=1
fi
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    export TENSOR_PARALLEL_SIZE=1
fi
if [ -z "$COMPLETION_MAX_PROMPT" ]; then
    export COMPLETION_MAX_PROMPT=14000
fi

echo "================================================"
echo "[MODEL_PATH] = ${MODEL_PATH}"
echo "[DATA_TYPE] = ${DATA_TYPE}"
echo "[MAX_NUM_SEQS] = ${MAX_NUM_SEQS}"
echo "[BLOCK_SIZE] = ${BLOCK_SIZE}"
echo "[GPU_MEMORY_UTILIZATION] = ${GPU_MEMORY_UTILIZATION}"
echo "[PIPELINE_PARALLEL_SIZE] = ${PIPELINE_PARALLEL_SIZE}"
echo "[TENSOR_PARALLEL_SIZE] = ${TENSOR_PARALLEL_SIZE}"
echo "[COMPLETION_MAX_PROMPT] = ${COMPLETION_MAX_PROMPT}"
echo "================================================"
pip list
echo "================================================"

python -m vllm.entrypoints.openai.api_server \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --seed 0 \
    --block-size ${BLOCK_SIZE} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --model ${MODEL_PATH} \
    --dtype ${DATA_TYPE} \
    -pp ${PIPELINE_PARALLEL_SIZE} \
    -tp ${TENSOR_PARALLEL_SIZE} \
    --host "0.0.0.0" \
    --port 8080