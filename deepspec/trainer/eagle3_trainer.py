from deepspec.data import CacheCollator
from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model
from deepspec.modeling.eagle3.gemma4.config import (
    build_draft_config as build_gemma4_eagle3_config,
)
from deepspec.modeling.eagle3.loss import compute_eagle3_loss
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
from deepspec.modeling.eagle3.qwen3.config import (
    build_draft_config as build_qwen3_eagle3_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3Eagle3Trainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def build_models(self):
        # Keep identical model construction to the base trainer so online
        # target-feature extraction uses the same frozen target checkpoint.
        return super().build_models()

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3Eagle3Model(draft_config)

    def run_batch(self, batch):
        if self.online:
            target_output = self._extract_online_target_hidden_states(batch)
            batch = dict(batch)
            batch["target_hidden_states"] = target_output.target_hidden_states
            batch["target_last_hidden_states"] = (
                target_output.target_last_hidden_states
            )
        return compute_eagle3_loss(
            model=self.model,
            batch=batch,
            ttt_length=int(self.draft_model.ttt_length),
            step_loss_decay=float(self.draft_model.step_loss_decay),
        )


class Gemma4Eagle3Trainer(Qwen3Eagle3Trainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4Eagle3Model(draft_config)
