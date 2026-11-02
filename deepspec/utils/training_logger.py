import time
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from deepspec.utils import ensure_dir, is_global_main_process, print_on_global_main
from deepspec.utils.metrics import add_metric, flush, reset


_writer: Optional[SummaryWriter] = None
_swanlab_enabled: bool = False
_logging_steps: int = 1
_session_start_wall: Optional[float] = None
_session_start_step: int = 0


def init(
    *,
    logging_steps: int,
    tensorboard_dir: Optional[str] = None,
    swanlab_config=None,
    run_name: Optional[str] = None,
    run_config=None,
) -> None:
    global _writer, _swanlab_enabled, _logging_steps
    _logging_steps = int(logging_steps)
    if tensorboard_dir is not None and is_global_main_process():
        ensure_dir(tensorboard_dir)
        _writer = SummaryWriter(tensorboard_dir)
    if swanlab_config and is_global_main_process():
        import os

        import swanlab

        mode = os.environ.get("SWANLAB_MODE") or swanlab_config.get(
            "mode", "online"
        )
        init_kwargs = dict(
            project=swanlab_config.get("project", "deepspec"),
            workspace=swanlab_config.get("workspace"),
            name=swanlab_config.get("name") or run_name,
            mode=mode,
            config=run_config,
        )
        log_dir = swanlab_config.get("dir") or swanlab_config.get("log_dir")
        if log_dir:
            init_kwargs["log_dir"] = log_dir
        try:
            swanlab.init(**init_kwargs)
            _swanlab_enabled = True
        except Exception as exc:
            # Do not take down a multi-hour training run if the tracker is
            # unreachable; keep a local SwanLab run instead.
            print_on_global_main(
                f"swanlab init failed in mode={mode!r} ({exc!r}); "
                "falling back to offline mode."
            )
            try:
                swanlab.init(mode="offline", **init_kwargs)
                _swanlab_enabled = True
            except Exception as fallback_exc:
                print_on_global_main(
                    f"swanlab offline fallback also failed ({fallback_exc!r}); "
                    "continuing without SwanLab."
                )
                _swanlab_enabled = False


def start_session(*, global_step: int) -> None:
    global _session_start_wall, _session_start_step
    reset()
    _session_start_wall = time.time()
    _session_start_step = int(global_step)


def on_optimizer_step(
    *,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
    learning_rate: float,
    grad_norm: float,
):
    add_metric("lr", learning_rate, reduction="last", tag="train")
    add_metric("grad_norm", grad_norm, reduction="last", tag="train")

    if global_step % _logging_steps != 0:
        return None

    summary = flush()
    if is_global_main_process():
        _write_scalars(summary, global_step=global_step)
        _print_summary(
            summary=summary,
            global_step=global_step,
            next_micro_step=next_micro_step,
            micro_batches_per_epoch=micro_batches_per_epoch,
            max_train_steps=max_train_steps,
        )
    return summary


def close() -> None:
    global _writer, _swanlab_enabled
    if _writer is not None:
        _writer.close()
        _writer = None
    if _swanlab_enabled:
        import swanlab

        swanlab.finish()
        _swanlab_enabled = False


def _swanlab_metric_name(name: str) -> str:
    """Group acceptance-related metrics under accept/ in SwanLab."""
    if name.startswith("train/accept_rate@"):
        return "accept/rate@" + name.rsplit("@", 1)[1]
    if name.startswith("train/accuracy@"):
        return "accept/accuracy@" + name.rsplit("@", 1)[1]
    if name in ("train/tau_probabilistic", "train/tau_greedy"):
        return "accept/" + name.split("/", 1)[1]
    return name


def _write_scalars(summary, *, global_step: int) -> None:
    if _writer is not None:
        for key, value in summary.items():
            _writer.add_scalar(key, value, global_step)
    if _swanlab_enabled:
        import swanlab

        swanlab.log(
            {_swanlab_metric_name(key): value for key, value in summary.items()},
            step=global_step,
        )


def _print_summary(
    *,
    summary,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
) -> None:
    session_start_wall = _session_start_wall
    if session_start_wall is None:
        session_start_wall = time.time()
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    session_elapsed = time.time() - session_start_wall
    completed_session_steps = global_step - _session_start_step
    remaining_steps = max(max_train_steps - global_step, 0)
    remaining_min = (
        session_elapsed * remaining_steps / max(completed_session_steps, 1)
    ) / 60
    loss_text = ""
    if "train/loss" in summary:
        loss_text = f" loss={summary['train/loss']:.4f}"
    print_on_global_main(
        f"epoch={current_epoch} "
        f"step={global_step}/{max_train_steps}"
        f"{loss_text} "
        f"| elapsed={session_elapsed / 60:.1f}min"
        f" | remaining={remaining_min:.1f}min"
    )
