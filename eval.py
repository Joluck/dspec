from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.eval.lightmlp import Qwen3LightMLPEvaluator
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
    "Qwen3LightMLPModel": Qwen3LightMLPEvaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of task names to evaluate (default: all tasks in TASKS).",
    )
    parser.add_argument(
        "--task-max-samples",
        type=int,
        default=None,
        help="Cap the number of samples per task (useful for quick checks).",
    )
    args = parser.parse_args()
    task_map = dict(TASKS)
    if args.tasks is None:
        args.tasks = list(TASKS)
    else:
        for name in args.tasks:
            if name in task_map:
                continue
            # Allow any JSONL that exists under eval_datasets/ as a task, e.g.
            # a held-out split produced by scripts/data/download_and_split.py.
            path = Path("eval_datasets") / f"{name}.jsonl"
            assert path.exists(), (
                f"Unknown eval task {name!r}: not in {sorted(task_map)} and "
                f"{path} does not exist."
            )
            task_map[name] = 500
        args.tasks = [(name, task_map[name]) for name in args.tasks]
    if args.task_max_samples is not None:
        assert args.task_max_samples > 0
        args.tasks = [
            (name, min(count, int(args.task_max_samples)))
            for name, count in args.tasks
        ]
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
