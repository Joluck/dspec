# DeepSpec

DeepSpec is a full-stack codebase for training and evaluating draft models for speculative decoding. It contains data preparation utilities, draft model implementations, training code, and evaluation scripts.

## Environment

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Data preparation additionally requires an inference engine to serve the target model when regenerating answers; see [scripts/data/README.md](./scripts/data/README.md) for details.

## Workflow

Run the stages in order — each stage's output feeds the next:

1. **Data Preparation** — download prompts and (optionally) regenerate target answers.
2. **Training** — train a draft model either online (frozen target model runs in the training loop) or offline (against a precomputed target cache).
3. **Evaluation** — measure speculative-decoding acceptance on benchmark tasks.

## Data Preparation

See [scripts/data/README.md](./scripts/data/README.md) for the step-by-step data pipeline:

1. download and split training data,
2. regenerate answers (recommended; online training also accepts the raw split),
3. *offline training only:* prepare the target cache (storage warning: this can be very large — roughly 38 TB for the default `Qwen/Qwen3-4B` setting).

Online training needs only the conversation JSONL from step 1/2. The target
model runs inside the training loop, so step 3 — and the 38 TB cache — can be
skipped entirely. Set `data.train_data_paths` to one or more conversation
JSONL files.

## Training

```bash
bash scripts/train/train.sh
```

`train.sh` launches `train.py`, which spawns one worker per visible GPU.
By default it runs **online** DSpark training for Qwen3-4B
([config/dspark/dspark_qwen3_4b_online.py](./config/dspark/dspark_qwen3_4b_online.py)):
each rank keeps a frozen copy of the target model, runs it once per micro-batch,
and feeds the captured hidden states to the draft model. This removes the
multi-terabyte target cache and lets you change the training data without
recomputing it.

```bash
# online (default)
bash scripts/train/train.sh

# choose another conversation JSONL / cap the dataset
TRAIN_DATA=/path/to/train.jsonl MAX_SAMPLES=100000 bash scripts/train/train.sh
```

To use the original offline path, point `CONFIG` at a cache config and pass the
cache directory explicitly:

```bash
CONFIG=config/dspark/dspark_qwen3_4b.py \
EXTRA_OPTS="--opts data.target_cache_path=${HOME}/.cache/deepspec/qwen3_4b_target_cache" \
bash scripts/train/train.sh
```

See the script header for the full list of configs and `--opts` overrides.
Checkpoints are written to `~/checkpoints/<project_name>/<exp_name>/step_*`.
Online checkpoints store a full fp32 optimizer state per rank, so the online
config keeps only the two newest ones (`logging.max_checkpoints_to_keep`).

Hardware: the published configs assume a single node with 8 GPUs. The online
Qwen3-4B config is sized for 4x 98GB GPUs (`local_batch_size=8`,
`num_anchors=128`, `max_length=2048`); lower `local_batch_size` or
`num_anchors`, or reduce `max_length`, on smaller cards.

### Equal-budget comparison (Eagle3 / DFlash / DSpark)

`scripts/train/compare_1000.sh` trains all three online draft algorithms for
the same **1000 optimizer steps**, on the same Qwen3-4B data, with the same
global batch of 64, and logs every run to Weights & Biases:

```bash
bash scripts/train/compare_1000.sh
```

The three runs are named `dspark_1000`, `dflash_1000`, and `eagle3_1000` in
project `deepspec` (workspace is set in each config's `logging.swanlab` block).
Eagle3 uses a per-GPU micro-batch of 4 because its 7-step TTT keeps more
activation memory; DFlash/DSpark use 8. The global batch stays 64 for all.

After training, evaluate the three checkpoints on identical held-out tasks:

```bash
bash scripts/eval/compare_1000_eval.sh
```

SwanLab is enabled per config via
`logging.swanlab = dict(project=..., workspace=..., name=..., mode="online")`.
Set `name=None` to use `exp_name` as the run name; set `SWANLAB_MODE=offline`
to keep the run local if the server is unreachable.

### LightMLP ablation (Eagle3 without attention)

`config/lightmlp/lightmlp_qwen3_4b_online.py` keeps Eagle3's target-hidden
projection, 7-step TTT objective and frozen embeddings/LM head, but replaces
the decoder layer's attention block with a `2h -> h` projection followed by the
Qwen3 MLP. There is no KV cache and no token mixing, so this is a per-position
MLP recurrence:

```bash
CONFIG=config/lightmlp/lightmlp_qwen3_4b_online.py \
bash scripts/train/train.sh
```

Its evaluator is registered as `Qwen3LightMLPModel`, so the normal
`eval.py` / `scripts/eval/eval.sh` flow works unchanged.


## Evaluation

```bash
bash scripts/eval/eval.sh
```

`eval.sh` runs `eval.py` against a trained draft checkpoint over the speculative-decoding benchmarks in [eval_datasets/](./eval_datasets/) (gsm8k, math500, aime25, humaneval, mbpp, livecodebench, mt-bench, alpaca, arena-hard-v2). Set:

- `target_name_or_path` — the target model the draft was trained against (e.g. `Qwen/Qwen3-4B`),
- `draft_name_or_path` — the draft checkpoint, e.g. `~/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest`, or one of the Hugging Face repo IDs listed in [Released Checkpoints](#released-checkpoints).

For a quick acceptance check, restrict the task list and sample count, e.g.:

```bash
python eval.py \
    --target_name_or_path /path/to/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_online_qwen3_4b/step_latest \
    --tasks gsm8k alpaca --task-max-samples 64 --max-new-tokens 512
```

### Released Checkpoints

The checkpoints below are the ones used for Table 1 in the [paper](https://arxiv.org/abs/2607.05147). Each checkpoint was trained on [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) data generated by its corresponding target model in non-thinking mode, and is the direct output of the corresponding training configuration under [config/](./config/).


| Algorithm | `Qwen/Qwen3-4B` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B` | `google/gemma-4-12B-it` |
| --- | --- | --- | --- | --- |
| Eagle3 | [deepseek-ai/eagle3_qwen3_4b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_4b_ttt7) | [deepseek-ai/eagle3_qwen3_8b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_8b_ttt7) | [deepseek-ai/eagle3_qwen3_14b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_14b_ttt7) | [deepseek-ai/eagle3_gemma4_12b_ttt7](https://huggingface.co/deepseek-ai/eagle3_gemma4_12b_ttt7) |
| DFlash | [deepseek-ai/dflash_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7) | [deepseek-ai/dflash_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_8b_block7) | [deepseek-ai/dflash_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_14b_block7) | [deepseek-ai/dflash_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dflash_gemma4_12b_block7) |
| DSpark | [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | [deepseek-ai/dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) |

> [!IMPORTANT]
> If you cite these results in a new paper, align your setup with the training settings in this repository; otherwise, the comparison is not meaningful. For domain-specific use, fine-tune the draft model again for better results, especially if the target model is expected to run in thinking mode.

## Supported Algorithms

Currently, DeepSpec includes three draft models: [DSpark](https://arxiv.org/abs/2607.05147), [DFlash](https://arxiv.org/abs/2602.06036) and [Eagle3](https://arxiv.org/abs/2503.01840).

## License

DeepSpec is released under the [MIT License](./LICENSE). It includes code adapted
from third-party projects under their own licenses; see [NOTICE](./NOTICE) for the
full attribution.

## Acknowledgements

DeepSpec builds on the ideas and code of several excellent open-source projects:

- [SpecForge](https://github.com/sgl-project/SpecForge) (Apache-2.0) — the overall training framework and Eagle3 implementation; portions of the Eagle3 modeling, loss, optimizer, attention, and evaluation code are adapted from it. Adapted files carry an in-file attribution comment, and the full notice is recorded in [NOTICE](./NOTICE).
- [DFlash](https://github.com/z-lab/dflash) (MIT) — the DFlash draft-model design and training recipe.
- [Qwen3](https://github.com/QwenLM/Qwen3) and [Gemma](https://github.com/google-deepmind/gemma) — the target model families supported in this repo.

We thank the authors and maintainers of these projects. Contributions of new algorithms are welcome.

## Citation

```bibtex
@misc{cheng2026dsparkconfidencescheduledspeculativedecoding,
      title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation}, 
      author={Xin Cheng and Xingkai Yu and Chenze Shao and Jiashi Li and Yunfan Xiong and Yi Qian and Jiaqi Zhu and Shirong Ma and Xiaokang Zhang and Jiasheng Ye and Qinyu Chen and Chengqi Deng and Jiping Yu and Damai Dai and Zhengyan Zhang and Yixuan Wei and Yixuan Tan and Wenkai Yang and Runxin Xu and Yu Wu and Zhean Xu and Xuanyu Wang and Muyang Chen and Rui Tian and Xiao Bi and Zhewen Hao and Shaoyuan Chen and Huanqi Cao and Wentao Zhang and Anyi Xu and Huishuai Zhang and Dongyan Zhao and Wenfeng Liang},
      year={2026},
      eprint={2607.05147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.05147}, 
}
```
