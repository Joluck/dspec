#!/usr/bin/env bash
# Evaluate the three equal-budget 1000-step runs on the same held-out tasks.
set -euo pipefail

cd "$(dirname "$0")/../.."
source /home/rwkv/jl/.mtp/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

TARGET=${TARGET_MODEL:-/home/rwkv/jl/models/Qwen3-4B}
TASKS=${TASKS:-"gsm8k alpaca perfectblend_60k_eval humaneval"}
MAX_SAMPLES=${MAX_SAMPLES:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

mkdir -p logs
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
    names=(dspark_1000 dflash_1000 eagle3_1000)
fi

for name in "${names[@]}"; do
    draft="${HOME}/checkpoints/deepspec/${name}/step_latest"
    if [ ! -e "${draft}" ]; then
        echo "missing checkpoint: ${draft}" >&2
        exit 1
    fi
    echo "============================================================"
    echo "[$(date '+%F %T')] evaluating ${name} (${draft})"
    echo "============================================================"
    python eval.py \
        --target_name_or_path "${TARGET}" \
        --draft_name_or_path "${draft}" \
        --tasks ${TASKS} \
        --task-max-samples "${MAX_SAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        2>&1 | tee "logs/eval_${name}.log"
done
