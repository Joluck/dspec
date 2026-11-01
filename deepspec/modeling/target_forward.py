"""Target-model hidden-state extraction shared by offline cache and online training.

The offline data pipeline captures a fixed set of target-layer hidden states and
writes them to a ``TargetCache`` on disk.  Online training performs exactly the
same forward pass inside the training loop instead, so both paths must use the
same hook placement and output-dtype conventions.  Keeping the implementation
here avoids the two paths drifting apart.
"""

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class TargetForwardResult:
    """Hidden states required to train a draft model.

    ``target_hidden_states`` concatenates the outputs of ``target_layer_ids``
    along the feature dimension, which is what DSpark/Eagle3 consume as context.
    ``target_last_hidden_states`` is the final backbone hidden state used by
    DSpark's distribution-matching loss and Eagle3's target logits.
    """

    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def get_target_backbone(target_model):
    """Return the text backbone that owns ``layers`` and ``embed_tokens``."""
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(
            target_model.model, "language_model"
        ):
            return target_model.model.language_model
        assert False, "Gemma4 target model must expose a text language_model."
    return getattr(target_model, "model", target_model)


def get_target_hidden_size(target_model) -> int:
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def validate_target_layer_ids_for_model(
    target_layer_ids: Iterable[int],
    target_model,
) -> list[int]:
    """Validate layer ids against the concrete backbone (including -1 = embedding)."""
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    assert target_layer_ids, "target_layer_ids must not be empty."
    num_layers = len(get_target_backbone(target_model).layers)
    previous = None
    for layer_id in target_layer_ids:
        assert layer_id == -1 or 0 <= layer_id < num_layers, (
            f"target_layer_id {layer_id} is out of range {{-1}} U "
            f"[0, {num_layers - 1}] for a target model with {num_layers} layers."
        )
        assert previous is None or layer_id > previous, (
            "target_layer_ids must be strictly increasing."
        )
        previous = layer_id
    return target_layer_ids


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
) -> TargetForwardResult:
    """Run the frozen target backbone and capture the requested layer states.

    The caller is responsible for putting ``target_model`` in eval mode and for
    freeing gradients; this helper itself always runs under ``torch.no_grad``.
    """
    backbone = get_target_backbone(target_model)
    layer_modules = backbone.layers
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured_hidden_states = {}
    handles = []

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            captured_hidden_states[layer_id] = _get_hook_tensor(output).detach()

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))
            )
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue
            handles.append(
                layer_modules[layer_id].register_forward_hook(capture_layer(layer_id))
            )

        with torch.no_grad():
            target_output = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            target_last_hidden_states = target_output.last_hidden_state
            missing = [
                layer_id
                for layer_id in target_layer_ids
                if layer_id not in captured_hidden_states
            ]
            assert not missing, f"Failed to capture target hidden states for {missing}."
            target_hidden_states = torch.cat(
                [captured_hidden_states[layer_id] for layer_id in target_layer_ids],
                dim=-1,
            )
    finally:
        for handle in handles:
            handle.remove()
        captured_hidden_states.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )


__all__ = [
    "TargetForwardResult",
    "get_target_backbone",
    "get_target_hidden_size",
    "run_target_forward_with_hooks",
    "validate_target_layer_ids_for_model",
]
