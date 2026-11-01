"""Dataset adapters for online (on-the-fly target-forward) draft training.

Offline training consumes a ``TargetCache`` containing precomputed target
hidden states.  Online training instead reads the same conversation JSONL that
would have been used to build the cache and runs the target model once per
micro-batch.  Only the conversation data is stored on disk in that mode.
"""

import os
from typing import Iterable

import torch

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.parser import preprocess_record
from deepspec.data.target_cache_dataset import ConversationCollator


def normalize_data_paths(data_paths) -> list[str]:
    if isinstance(data_paths, (str, os.PathLike)):
        data_paths = [data_paths]
    paths = [os.path.abspath(os.path.expanduser(str(path))) for path in data_paths]
    assert paths, "data.train_data_paths must contain at least one JSONL path."
    for path in paths:
        assert os.path.isfile(path), f"Online training JSONL does not exist: {path}"
    return paths


class OnlineJsonDataset(torch.utils.data.Dataset):
    """Conversation JSONL dataset with an optional deterministic prefix cap.

    ``max_samples`` is applied before distributed sharding so every rank sees
    the same global sample set and the resumable sampler remains deterministic.
    Set it to ``None`` to train on every record in the JSONL file(s).
    """

    def __init__(self, data_paths: Iterable[str] | str, max_samples: int | None = None):
        super().__init__()
        self.data_paths = normalize_data_paths(data_paths)
        self.max_samples = None if max_samples is None else int(max_samples)
        if self.max_samples is not None:
            assert self.max_samples > 0, "data.max_samples must be positive."
        self._jsonl_dataset = JsonLineDataset(data_paths=self.data_paths)
        total = len(self._jsonl_dataset)
        assert total > 0, f"Online training JSONL has no records: {self.data_paths}"
        self.num_samples = total if self.max_samples is None else min(
            total, self.max_samples
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        return self._jsonl_dataset[int(index)]

    def close(self):
        self._jsonl_dataset.close()

    def __del__(self):  # pragma: no cover
        self.close()


class OnlineConversationCollator(ConversationCollator):
    """Conversation collator that never returns ``None``.

    The offline cache builder drops samples with too few supervised tokens.  A
    training loop cannot skip a micro-batch without changing the optimizer
    schedule, so online mode instead keeps the shape valid and zeroes the loss
    mask when every sample in the batch was filtered out.
    """

    def __call__(self, features):
        batch = super().__call__(features)
        if batch is not None:
            return batch
        processed = preprocess_record(
            record=features[0],
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            max_length=self.max_length,
        )
        processed["loss_mask"] = torch.zeros_like(processed["loss_mask"])
        batch_size = len(features)
        return {
            key: processed[key].unsqueeze(0).repeat(batch_size, 1)
            for key in ("input_ids", "attention_mask", "loss_mask")
        }


__all__ = [
    "OnlineConversationCollator",
    "OnlineJsonDataset",
    "normalize_data_paths",
]
