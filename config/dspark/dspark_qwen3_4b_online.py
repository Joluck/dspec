import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

# DSpark is DeepSpec's multi-token-prediction draft: each anchor predicts a
# block of `block_size` (7) tokens, with a Markov head for intra-block
# dependencies and a confidence head for early stopping.  This config trains it
# online against a frozen local Qwen3-4B checkpoint.
project_name = "deepspec"
exp_name = "dspark_online_qwen3_4b"
seed = 42

# Local checkpoint used for the Qwen3-4B MTP run.  Override with
# --opts "model.target_model_name_or_path=<path-or-hf-id>".
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
    # Fewer anchors than the offline recipe: online training runs the 4B
    # target in the loop and the DSpark logits tensor scales with
    # batch * anchors * block_size * vocab. 128 anchors x 7 slots still gives
    # 896 supervised draft tokens per sample and fits 8 samples/GPU here.
    num_anchors=128,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    # 4x RTX PRO 6000 (98GB) fits 8 samples per rank comfortably.  Lower it
    # to 1-4 on smaller cards.
    local_batch_size=8,
    # The offline recipe uses 512.  With a ~60k-sample online subset, 128 gives
    # ~2300 optimizer steps (vs ~580 at 512) in the same wall-clock time, which
    # matters much more for a small-data run.  Raise this back to 512 when
    # training on the full 1.4M-sample split.
    global_batch_size=128,
    # 5 epochs x ~59k samples at global batch 128 is ~2300 optimizer steps.
    # The full offline recipe uses 10 epochs over 1.4M samples (~27k steps).
    num_train_epochs=5,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=300,
    # Online checkpoints carry a full fp32 optimizer state per rank (~31GB for
    # 4 ranks). Keep only the two newest so long runs do not fill the disk.
    max_checkpoints_to_keep=2,
    wandb=dict(
        project="deepspec",
        entity="jl-ai",
        name=None,
        mode="online",
    ),
)

# Online training: no target cache.  The frozen Qwen3-4B target model runs
# once per micro-batch to produce the hidden states consumed by DSpark.
data = dict(
    online=True,
    # Any conversation JSONL works. This 60k split is produced by
    # scripts/data/download_and_split.py --sample-size 60000; the canonical
    # full-size alternative is train_datasets/qwen3_4b/perfectblend_train_regen.jsonl.
    train_data_paths=["train_datasets/perfectblend_train_60k.jsonl"],
    max_samples=None,
    min_loss_tokens=14,
    chat_template="qwen",
    # 99% of open-perfectblend samples are shorter than 2.1k tokens; capping
    # here keeps online target-forward memory predictable. Raise to 4096 (and
    # lower local_batch_size) for long-context runs.
    max_length=2048,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
