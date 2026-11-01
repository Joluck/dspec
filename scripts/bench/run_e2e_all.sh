#!/usr/bin/env bash
# Baseline vs four speculative decoders on identical prompts.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/rwkv/jl/.mtp/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

TASKS=${TASKS:-"gsm8k alpaca humaneval perfectblend_60k_eval"}
SAMPLES=${SAMPLES:-32}
MAX_NEW=${MAX_NEW:-256}

mkdir -p logs artifacts
for method in baseline dspark dflash eagle3 lightmlp; do
    echo "============================================================"
    echo "[$(date '+%F %T')] e2e ${method}"
    echo "============================================================"
    python scripts/bench/bench_end_to_end.py \
        --method "${method}" \
        --tasks ${TASKS} \
        --samples-per-task "${SAMPLES}" \
        --max-new-tokens "${MAX_NEW}" \
        --output "artifacts/e2e_${method}.json" \
        2>&1 | tee "logs/e2e_${method}.log"
done
echo "done"
