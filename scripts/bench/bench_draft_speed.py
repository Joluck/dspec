"""Draft-only proposal latency for the four 1000-step checkpoints.

Measures the time to produce one 7-token draft proposal at batch=1 and a
1024-token context, excluding target-model verification.  DSpark/DFlash use a
single block forward; Eagle3/LightMLP use the evaluator's 7-step TTT loop.
"""

import argparse
import json
import os
import statistics
import time

import torch
from transformers import DynamicCache

from deepspec.eval.dspark.draft_ops import (
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
from deepspec.modeling.lightmlp.qwen3 import Qwen3LightMLPModel
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.utils.sampling import logits_to_probs, sample_tokens


def _cuda_time(fn, warmup=3, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


@torch.inference_mode()
def bench_dspark(path, ctx, block):
    model = Qwen3DSparkModel.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    h = int(model.config.hidden_size)
    n = len(model.target_layer_ids)
    target_hidden = torch.randn(1, ctx, n * h, device="cuda", dtype=torch.bfloat16)
    draft_ids = torch.full(
        (1, block), int(model.mask_token_id), device="cuda", dtype=torch.long
    )
    draft_ids[:, 0] = 100
    position_ids = torch.arange(ctx + block, device="cuda").unsqueeze(0)

    def one():
        cache = DynamicCache()
        block_hidden = forward_dspark_draft_block(
            model,
            draft_input_ids=draft_ids,
            position_ids=position_ids,
            past_key_values_draft=cache,
            target_hidden_states=target_hidden,
            start=ctx,
            block_size=block,
        )
        return build_dspark_proposal(
            model=model,
            draft_input_ids=draft_ids,
            block_hidden=block_hidden,
            block_size=block,
            temperature=1.0,
            confidence_threshold=0.0,
        )

    return _cuda_time(one)


@torch.inference_mode()
def bench_eagle_like(path, model_cls, ctx, block):
    model = model_cls.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    h = int(model.config.hidden_size)
    n = len(model.target_layer_ids)
    target_hidden = torch.randn(1, ctx, n * h, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(1, int(model.config.vocab_size), (1, ctx), device="cuda")
    position_ids = torch.arange(ctx, device="cuda").unsqueeze(0)
    base_cache = DynamicCache()
    context_hidden = model.extend_draft_cache(
        hidden_states=target_hidden,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=base_cache,
    )

    def one():
        cache = base_cache
        cache.crop(ctx)
        last_hidden = context_hidden
        logits = []
        for step in range(block):
            step_logits = model.compute_logits(last_hidden)
            logits.append(step_logits)
            next_token = sample_tokens(step_logits, temperature=1.0)
            if step + 1 < block:
                last_hidden = model(
                    hidden_states=last_hidden,
                    input_ids=next_token[:, -1:],
                    position_ids=position_ids[:, :1],
                    past_key_values=cache,
                    use_cache=True,
                )
        draft_logits = torch.cat(logits, dim=1)
        return logits_to_probs(draft_logits, 1.0)

    return _cuda_time(one)


def main():
    root = os.path.expanduser("~/checkpoints/deepspec")
    specs = [
        ("DSpark", "dspark_1000", "dspark"),
        ("DFlash", "dflash_1000", "dspark"),
        ("Eagle3", "eagle3_1000", "eagle3"),
        ("LightMLP", "lightmlp_1000", "lightmlp"),
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--output", default="artifacts/draft_speed.json")
    args = parser.parse_args()
    results = {}
    for label, run, kind in specs:
        path = f"{root}/{run}/step_1000"
        if kind == "dspark":
            median_ms, min_ms = bench_dspark(path, args.context_length, args.block_size)
        elif kind == "eagle3":
            median_ms, min_ms = bench_eagle_like(
                path, Qwen3Eagle3Model, args.context_length, args.block_size
            )
        else:
            median_ms, min_ms = bench_eagle_like(
                path, Qwen3LightMLPModel, args.context_length, args.block_size
            )
        results[label] = {
            "median_ms_per_proposal": round(median_ms, 3),
            "min_ms_per_proposal": round(min_ms, 3),
            "proposals_per_s": round(1000.0 / median_ms, 2),
            "draft_tokens_per_s": round(args.block_size * 1000.0 / median_ms, 1),
        }
        print(label, results[label], flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
