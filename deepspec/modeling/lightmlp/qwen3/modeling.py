"""Eagle3-style multi-step draft head with the attention block removed.

At each TTT step the layer consumes the previous step's hidden state and the
current token embedding, projects their concatenation back to the model width,
and applies the Qwen3 MLP.  There is no token mixing (no QKV / attention / KV
cache), so the model is a per-position MLP recurrence.  The remaining interface
(``fc`` projection, ``ttt_length``, ``compute_logits``, ``extend_draft_cache``)
matches Eagle3 so the existing online trainer, loss and evaluator can be reused.
"""

from typing import Optional

import torch
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
)

from deepspec.modeling.eagle3.common import Eagle3ForwardOutput
from deepspec.utils.sampling import sample_tokens


class Qwen3LightMLPLayer(nn.Module):
    """MLP-only replacement for the Eagle3 decoder layer's attention block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Replaces q/k/v/o attention with a single 2h -> h projection.
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(config)
        self.layer_idx = int(layer_idx)

    def forward(
        self,
        *,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.proj(
            torch.cat((input_embeds, hidden_states), dim=-1)
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3LightMLPModel(Qwen3PreTrainedModel):
    _no_split_modules = ["Qwen3LightMLPLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = ("target_layer_ids", "ttt_length", "step_loss_decay")
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        self.ttt_length = int(config.ttt_length)
        self.step_loss_decay = float(config.step_loss_decay)

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3LightMLPLayer(config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == (
            len(self.target_layer_ids) * self.config.hidden_size
        )
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def draft_sample(
        self, logits: torch.Tensor, temperature: float = 0.0
    ) -> torch.Tensor:
        return sample_tokens(logits, temperature=temperature)

    def extend_draft_cache(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
    ) -> torch.Tensor:
        """Return the hidden state for the last input position.

        Kept for Eagle3 evaluator compatibility.  There is no KV cache to
        extend; ``past_key_values``/``position_ids`` are accepted and ignored.
        """
        del position_ids, past_key_values
        output = self(
            hidden_states=hidden_states,
            input_ids=input_ids,
            use_cache=False,
        )
        return output[:, -1:, :]

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        target_logits_only: bool = False,
        return_logits: bool = False,
        rope_cache_step_offset: bool = False,
        **kwargs,
    ) -> torch.Tensor | Eagle3ForwardOutput:
        del (
            position_ids,
            past_key_values,
            use_cache,
            attention_mask,
            rope_cache_step_offset,
            kwargs,
        )
        if target_logits_only:
            assert target_last_hidden_states is not None
            with torch.no_grad():
                return self.lm_head(target_last_hidden_states)

        assert hidden_states is not None, "hidden_states must be provided."
        if hidden_states.size(-1) == (
            len(self.target_layer_ids) * self.config.hidden_size
        ):
            hidden_states = self.project_hidden_states(hidden_states)

        if input_embeds is None:
            assert input_ids is not None, (
                "Either input_ids or input_embeds must be provided."
            )
            input_embeds = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                input_embeds=input_embeds,
                hidden_states=hidden_states,
            )

        if return_logits:
            draft_logits = self.compute_logits(hidden_states)
            target_logits = None
            if target_last_hidden_states is not None:
                with torch.no_grad():
                    target_logits = self.lm_head(target_last_hidden_states)
            return Eagle3ForwardOutput(
                hidden_states=hidden_states,
                draft_logits=draft_logits,
                target_logits=target_logits,
            )
        return hidden_states


__all__ = ["Qwen3LightMLPModel", "Qwen3LightMLPLayer"]
