#!/usr/bin/env bash

# Local launch mirrors the repo's node launcher, not standard
# torchrun semantics. train.py spawns one worker per visible GPU by itself.
# init_dist defaults to single-node local run when MASTER_ADDR/MASTER_PORT and
# RANK/WORLD_SIZE are unset. Total GPU workers come from CUDA_VISIBLE_DEVICES.

# Default: online DSpark training for Qwen3-4B. The target model runs inside
# the training loop, so no multi-terabyte target cache is needed.
#
# Offline training is still available by pointing CONFIG at one of the cache
# configs and passing data.target_cache_path, e.g.:
#   CONFIG=config/dspark/dspark_qwen3_4b.py \
#   EXTRA_OPTS='--opts data.target_cache_path=...' bash scripts/train/train.sh
#
# Available public configs:
## dspark
#   config/dspark/dspark_qwen3_4b_online.py
#   config/dspark/dspark_qwen3_4b.py
#   config/dspark/dspark_qwen3_8b.py
#   config/dspark/dspark_qwen3_14b.py
#   config/dspark/dspark_gemma4_12b.py
## dflash
#   config/dflash/dflash_qwen3_4b.py
#   config/dflash/dflash_qwen3_8b.py
#   config/dflash/dflash_qwen3_14b.py
#   config/dflash/dflash_gemma4_12b.py
## lightmlp (Eagle3 without attention)
#   config/lightmlp/lightmlp_qwen3_4b_online.py
## eagle3
#   config/eagle3/eagle3_qwen3_4b.py
#   config/eagle3/eagle3_qwen3_8b.py
#   config/eagle3/eagle3_qwen3_14b.py
#   config/eagle3/eagle3_gemma4_12b.py

CONFIG=${CONFIG:-config/dspark/dspark_qwen3_4b_online.py}
TRAIN_DATA=${TRAIN_DATA:-train_datasets/perfectblend_train_60k.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-}
EXTRA_OPTS=${EXTRA_OPTS:-}

# Only inject online-data overrides for configs that opt into online training;
# offline cache configs do not define data.train_data_paths.
opts=()
if grep -Eq "online[[:space:]]*=[[:space:]]*True" "${CONFIG}"; then
    opts+=(--opts "data.train_data_paths=[\"${TRAIN_DATA}\"]")
    if [ -n "${MAX_SAMPLES}" ]; then
        opts+=(--opts "data.max_samples=${MAX_SAMPLES}")
    fi
fi

# --opts overrides any config field by dotted key path. local_batch_size is
# the per-GPU micro-batch size; lower it to 1-4 on smaller cards, or override
# it without editing the config via:
#   EXTRA_OPTS='--opts train.local_batch_size=4' bash scripts/train/train.sh
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} python train.py \
    --config "${CONFIG}" \
    "${opts[@]}" \
    ${EXTRA_OPTS}
