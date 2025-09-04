set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


MODEL_PATH=/lpai/volumes/base-mindgpt-ali-sh-mix/luhengtong/base_models/Qwen_models/Qwen3/Qwen3-8B

DATA_DIR=/lpai/volumes/base-mindgpt-ali-sh-mix/libingwei/asearch/ASearcher-test-data

OUTPUT_DIR=/lpai/volumes/base-mindgpt-ali-sh-mix/libingwei/asearch/outputs

# 设置CUDA_VISIBLE_DEVICES环境变量，指定可用的GPU
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \

SPLIT=1
MAX_GEN_TOKENS=4098
DATA_NAMES=GAIA
AGENT_TYPE=asearcher-reasoning
PROMPT_TYPE=asearcher-reasoning
SEARCH_CLIENT_TYPE=async-web-search-access
temperature=0.6
top_p=0.95
top_k=-1

echo "MODEL PATH: $MODEL_PATH"
echo "Temperature: ${temperature}"
echo "top_p: ${top_p}"
echo "top_k: ${top_k}"
echo "split: ${SPLIT}/"

TOKENIZERS_PARALLELISM=false \
PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --data_dir ${DATA_DIR} \
    --split test \
    --search-client-type ${SEARCH_CLIENT_TYPE} \
    --max-tokens-per-call ${MAX_GEN_TOKENS} \
    --tensor_parallel_size 1 \
    --num_test_sample -1 \
    --n_sampling 1 \
    --temperature ${temperature} \
    --top_p $top_p \
    --top_k $top_k \
    --start 0 \
    --end -1 \
    --seed 1 \
    --parallel-mode seed \
    --llm_as_judge \
    --pass-at-k 2 \
