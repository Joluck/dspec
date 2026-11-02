import os

from deepspec.trainer import Qwen3Eagle3Trainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "deepspec"
exp_name = "eagle3_online_qwen3_4b"
seed = 0

_LOCAL_QWEN3_4B = "/home/rwkv/jl/models/Qwen3-4B"
TARGET_MODEL_PATH = os.environ.get("DEEPSPEC_TARGET_MODEL") or (
    _LOCAL_QWEN3_4B if os.path.isdir(_LOCAL_QWEN3_4B) else "Qwen/Qwen3-4B"
)

model = dict(
    target_model_name_or_path=TARGET_MODEL_PATH,
    target_layer_ids=[1, 9, 17, 25, 33],
    ttt_length=7,
    step_loss_decay=0.8,
    draft_num_hidden_layers=1,
)

train = dict(
    trainer_cls=Qwen3Eagle3Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    # Eagle3 TTT keeps 7 draft-logit tensors per micro-batch; 2 is the safe
    # value on 95GB cards at max_length=2048.
    local_batch_size=2,
    global_batch_size=64,
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
    swanlab=dict(
        project="deepspec",
        workspace="joluck",
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
