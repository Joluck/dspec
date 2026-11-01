import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "deepspec"
exp_name = "dflash_online_qwen3_4b"
seed = 42

_LOCAL_QWEN3_4B = "/home/rwkv/jl/models/Qwen3-4B"
TARGET_MODEL_PATH = os.environ.get("DEEPSPEC_TARGET_MODEL") or (
    _LOCAL_QWEN3_4B if os.path.isdir(_LOCAL_QWEN3_4B) else "Qwen/Qwen3-4B"
)

model = dict(
    target_model_name_or_path=TARGET_MODEL_PATH,
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=128,

    # DFlash = DSpark backbone without the Markov / confidence heads.
    markov_rank=0,
    confidence_head_alpha=0.0,

    # CE-only loss.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=8,
    global_batch_size=128,
    num_train_epochs=5,
    max_train_steps=1000,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=250,
    max_checkpoints_to_keep=2,
    wandb=dict(
        project="deepspec",
        entity="jl-ai",
        name=None,
        mode="online",
    ),
)

data = dict(
    online=True,
    train_data_paths=["train_datasets/perfectblend_train_60k.jsonl"],
    max_samples=None,
    min_loss_tokens=14,
    chat_template="qwen",
    max_length=2048,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR, project_name, exp_name
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR, project_name, exp_name
    )
    cfg["logging"] = logging_cfg
    return cfg
