from deepspec.modeling.lightmlp.qwen3 import Qwen3LightMLPModel
from deepspec.modeling.lightmlp.qwen3.config import (
    build_draft_config as build_qwen3_lightmlp_config,
)
from deepspec.trainer.eagle3_trainer import Qwen3Eagle3Trainer


class Qwen3LightMLPTrainer(Qwen3Eagle3Trainer):
    """Eagle3 trainer with the attention-free LightMLP draft backbone."""

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_lightmlp_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3LightMLPModel(draft_config)


__all__ = ["Qwen3LightMLPTrainer"]
