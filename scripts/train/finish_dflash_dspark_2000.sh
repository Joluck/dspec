#!/usr/bin/env bash
# Finish the 2000-step comparison: DFlash and DSpark, same global batch 64
# and data as lightmtp_2000. LightMLP/MTP is already trained.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/rwkv/jl/.mtp/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

DATA=/home/rwkv/jl/DeepSpec/train_datasets/perfectblend_train_60k.jsonl
mkdir -p logs

run_one() {
    local name="$1"
    local config="$2"
    echo "============================================================"
    echo "[$(date '+%F %T')] starting ${name}"
    echo "============================================================"
    python train.py \
        --config "${config}" \
        --opts "exp_name=${name}" \
        --opts train.max_train_steps=2000 \
        --opts train.local_batch_size=8 \
        --opts train.global_batch_size=64 \
        --opts "data.train_data_paths=[\"${DATA}\"]" \
        --opts logging.logging_steps=10 \
        --opts logging.checkpointing_steps=500 \
        --opts logging.max_checkpoints_to_keep=1 \
        2>&1 | tee "logs/${name}.log"
    echo "[$(date '+%F %T')] finished ${name}"
}

run_one dflash_2000 config/dflash/dflash_qwen3_4b_online.py
run_one dspark_2000 config/dspark/dspark_qwen3_4b_online.py
echo "done"
