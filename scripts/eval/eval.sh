# Local launch mirrors the repo's node launcher, not standard torchrun
# semantics. eval.py spawns one worker per visible GPU by itself.
# init_dist defaults to single-node local run when MASTER_ADDR/MASTER_PORT and
# RANK/WORLD_SIZE are unset. Total GPU workers come from CUDA_VISIBLE_DEVICES.

# Match this to the target model used by the draft checkpoint.
if [ -z "${TARGET_MODEL:-}" ]; then
    if [ -d /home/rwkv/jl/models/Qwen3-4B ]; then
        TARGET_MODEL=/home/rwkv/jl/models/Qwen3-4B
    else
        TARGET_MODEL=Qwen/Qwen3-4B
    fi
fi
target_name_or_path=${TARGET_MODEL}

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
# Use step_latest for the most recent checkpoint, or replace it with step_<N>.
draft_name_or_path=${DRAFT_MODEL:-${HOME}/checkpoints/deepspec/dspark_online_qwen3_4b/step_latest}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} python eval.py \
    --target_name_or_path ${target_name_or_path} \
    --draft_name_or_path ${draft_name_or_path}
