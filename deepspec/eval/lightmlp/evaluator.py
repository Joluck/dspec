from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator
from deepspec.modeling.lightmlp.qwen3 import Qwen3LightMLPModel


class Qwen3LightMLPEvaluator(Qwen3Eagle3Evaluator):
    """Eagle3 evaluator with the attention-free LightMLP draft model."""

    draft_model_cls = Qwen3LightMLPModel


__all__ = ["Qwen3LightMLPEvaluator"]
