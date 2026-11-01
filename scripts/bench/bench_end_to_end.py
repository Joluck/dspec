"""End-to-end speculative-decoding benchmark against a plain target baseline.

For each method we run the same prompts, temperature, stop tokens and
max_new_tokens.  Baseline uses a plain target-only incremental loop; speculative
methods use their evaluator's ``generate_one_sample``.  Timing excludes model
loading but includes prefill + all decode steps.
"""

import argparse
import json
import os
import random
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from deepspec.eval.base_evaluator import (
    encode_chat_messages,
    has_stop_token,
    load_and_process_dataset,
    resolve_stop_token_ids,
    trim_output_ids,
)
from deepspec.eval.dspark import Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator
from deepspec.eval.lightmlp import Qwen3LightMLPEvaluator
from deepspec.utils import init_dist
from deepspec.utils.sampling import logits_to_probs, sample_from_probs

METHODS = {
    "dspark": ("dspark_1000", Qwen3DSparkEvaluator),
    "dflash": ("dflash_1000", Qwen3DSparkEvaluator),
    "eagle3": ("eagle3_1000", Qwen3Eagle3Evaluator),
    "lightmlp": ("lightmlp_1000", Qwen3LightMLPEvaluator),
}


@torch.inference_mode()
def baseline_generate(
    model,
    input_ids,
    *,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids,
):
    """Plain target-only generation, one target forward per output token."""
    device = input_ids.device
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)
    position_ids = torch.arange(max_length + 1, device=device).unsqueeze(0)
    past_key_values = DynamicCache()
    output = model(
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    output_ids = torch.empty(
        (1, max_length + 1), dtype=torch.long, device=device
    )
    output_ids[:, :num_input_tokens] = input_ids
    next_token = sample_from_probs(
        logits_to_probs(output.logits, float(temperature))
    )
    output_ids[:, num_input_tokens : num_input_tokens + 1] = next_token
    start = num_input_tokens
    if has_stop_token(next_token, stop_token_ids):
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=int(output_ids.shape[1] - num_input_tokens),
            acceptance_lengths=[],
            verify_count=0,
        )

    while start < max_length - 1:
        output = model(
            input_ids=output_ids[:, start : start + 1],
            position_ids=position_ids[:, start : start + 1],
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = sample_from_probs(
            logits_to_probs(output.logits, float(temperature))
        )
        start += 1
        output_ids[:, start : start + 1] = next_token
        if has_stop_token(next_token, stop_token_ids):
            break

    output_ids = trim_output_ids(output_ids[:, : start + 1], num_input_tokens, stop_token_ids)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=int(output_ids.shape[1] - num_input_tokens),
        acceptance_lengths=[],
        verify_count=0,
    )


def build_prompts(args, tokenizer, device):
    rows = []
    for task in args.tasks:
        dataset = load_and_process_dataset(task)
        rng = random.Random(int(args.seed))
        rng.shuffle(dataset)
        rows.extend(dataset[: int(args.samples_per_task)])
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": row["turns"][0]}]
        prompts.append(
            encode_chat_messages(
                tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(device)
        )
    return prompts


def main(local_rank: int, args):
    if args.method == "baseline":
        device, global_rank, world_size = init_dist(local_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.target_name_or_path)
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)
        generate = lambda ids, n: baseline_generate(
            target_model,
            ids,
            max_new_tokens=n,
            temperature=float(args.temperature),
            stop_token_ids=stop_token_ids,
        )
    else:
        run_name, evaluator_cls = METHODS[args.method]
        draft_path = args.draft_name_or_path or os.path.expanduser(
            f"~/checkpoints/deepspec/{run_name}/step_latest"
        )
        eval_args = SimpleNamespace(
            target_name_or_path=args.target_name_or_path,
            draft_name_or_path=draft_path,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            confidence_threshold=0.0,
            tensorboard_dir=None,
            step=None,
            seed=int(args.seed),
            tasks=[],
        )
        evaluator = evaluator_cls(local_rank, eval_args)
        # The DSpark evaluator attaches a calibration recorder only for the
        # offline eval pipeline; it is not part of normal decoding.
        if getattr(evaluator, "confidence_head_recorder", None) is not None:
            evaluator.confidence_head_recorder = None
        device = evaluator.device
        global_rank = evaluator.global_rank
        world_size = evaluator.world_size
        tokenizer = evaluator.tokenizer
        stop_token_ids = resolve_stop_token_ids(
            evaluator.target_model, evaluator.tokenizer
        )
        generate = lambda ids, n: evaluator.generate_one_sample(
            input_ids=ids, stop_token_ids=stop_token_ids
        )

    prompts = build_prompts(args, tokenizer, device)
    # Warm up kernels / allocator on one prompt and exclude it from timing.
    if prompts:
        generate(prompts[0], 8)
        torch.cuda.synchronize(device)

    total_time = 0.0
    total_tokens = 0
    total_samples = 0
    accept_num = 0.0
    accept_den = 0.0
    proposal_count = 0

    for idx in range(global_rank, len(prompts), world_size):
        input_ids = prompts[idx]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        response = generate(input_ids, int(args.max_new_tokens))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_tokens += int(response.num_output_tokens)
        total_samples += 1
        if args.method == "baseline":
            # One target verification per emitted token.
            accept_num += float(response.num_output_tokens)
            accept_den += float(response.num_output_tokens)
        else:
            lengths = list(response.acceptance_lengths)
            accept_num += float(sum(lengths))
            accept_den += float(len(lengths))
            proposal_count += int(response.verify_count)

    stats = torch.tensor(
        [
            total_time,
            float(total_tokens),
            float(total_samples),
            accept_num,
            accept_den,
            float(proposal_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_time, total_tokens, total_samples, accept_num, accept_den, proposal_count = (
        stats.tolist()
    )
    result = {
        "method": args.method,
        "target": args.target_name_or_path,
        "samples": int(total_samples),
        "output_tokens": int(total_tokens),
        "seconds": total_time,
        "tokens_per_s": total_tokens / max(total_time, 1e-9),
        "ms_per_sample": 1000.0 * total_time / max(total_samples, 1.0),
        "mean_accept_len": accept_num / max(accept_den, 1.0),
        "proposals": int(proposal_count),
        "tokens_per_proposal": total_tokens / max(proposal_count, 1.0)
        if args.method != "baseline"
        else 1.0,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "tasks": list(args.tasks),
        "samples_per_task": int(args.samples_per_task),
    }
    if global_rank == 0:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["baseline", *METHODS.keys()],
    )
    parser.add_argument(
        "--target-name-or-path", default="/home/rwkv/jl/models/Qwen3-4B"
    )
    parser.add_argument("--draft-name-or-path", default=None)
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "alpaca"])
    parser.add_argument("--samples-per-task", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--output", default="artifacts/e2e_baseline.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
