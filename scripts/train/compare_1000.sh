#!/usr/bin/env bash
# Train Eagle3 / DFlash / DSpark for an equal-budget 1000 optimizer steps on
# the same online Qwen3-4B data, logging all three runs to W&B.
#
# Shared budget:
#   * global batch 64, 1000 optimizer steps (~64k samples seen)
#   * online target forward, max_length 2048, open-perfectblend 60k split
#   * lr 6e-4, warmup 4%, bf16, 4 GPUs
# Eagle3 needs a smaller per-GPU micro-batch (4 vs 8) because its 7-step TTT
# keeps more activation memory; global batch is identical for all three.
set -euo pipefail

cd "$(dirname "$0")/../.."
source /home/rwkv/jl/.mtp/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

run_one() {
    local name="$1"
    local config="$2"
    local local_bs="$3"
    echo "============================================================"
    echo "[$(date '+%F %T')] starting ${name}: ${config} (local_bs=${local_bs})"
    echo "============================================================"
    python train.py \
        --config "${config}" \
        --opts "exp_name=${name}" \
        --opts train.max_train_steps=1000 \
        --opts train.local_batch_size="${local_bs}" \
        --opts train.global_batch_size=64 \
        --opts logging.logging_steps=10 \
        --opts logging.checkpointing_steps=500 \
        --opts logging.max_checkpoints_to_keep=1 \
        2>&1 | tee "logs/compare_${name}.log"
    echo "[$(date '+%F %T')] finished ${name}"
}

# Order: shortest/most-fragile first so failures surface early.
run_one dspark_1000 config/dspark/dspark_qwen3_4b_online.py 8
run_one dflash_1000 config/dflash/dflash_qwen3_4b_online.py 8
run_one eagle3_1000 config/eagle3/eagle3_qwen3_4b_online.py 2

echo "All three 1000-step runs finished."
