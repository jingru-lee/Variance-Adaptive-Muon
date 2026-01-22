import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
import yaml

from logger.logger import DynamicsLogger
from optim.weight_averaging import (ExponentialWeightAverager, WeightAverager,
                                    eval_ewa, eval_wa)

from .utils import (eval, get_batch, get_parameter_norms, load_checkpoint,
                    load_worker_state, save_checkpoint,
                    save_worker_state, visualize_routing)


def setup_wandb_time_metrics():
    """
    Configure wandb metrics for plotting loss vs training time curves.
    Uses seconds as the time unit for the x-axis.
    """
    # Define metrics with training time (seconds) as x-axis
    wandb.define_metric("loss_vs_time/train_time_seconds")
    wandb.define_metric("loss_vs_time/train_loss", step_metric="loss_vs_time/train_time_seconds")
    wandb.define_metric("loss_vs_time/val_loss", step_metric="loss_vs_time/train_time_seconds")


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if cfg.weight_average:
        weight_averager = WeightAverager(
            save_dir=exp_dir / "weight_averages",
            interval=cfg.wa_interval,
            horizon=cfg.wa_horizon,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            use_temp_dir=cfg.wa_use_temp_dir,
        )

    if cfg.exponential_weight_average:
        ewa = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ewa_interval,
            decay=cfg.ewa_decay,
            warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    # Setup wandb time-related metrics (only on master process with wandb enabled)
    if distributed_backend.is_master_process() and cfg.wandb:
        setup_wandb_time_metrics()

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)

    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    grad_norms = []
    model.train()

    # Record training start time and cumulative training time
    training_start_time = time.time()
    cumulative_train_time = 0.0  # Cumulative training time in seconds (excluding evaluation time)

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                tokens,
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
                cumulative_train_time=cumulative_train_time,
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

            if cfg.exponential_weight_average:
                eval_ewa(
                    curr_iter,
                    not_compiled_model,
                    ewa,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        # Synchronize GPU before starting timer
        if "cuda" in cfg.device:
            torch.cuda.synchronize()
        t_start = time.perf_counter_ns()
        
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y, get_logits=False, moe=cfg.moe)
            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            grad_norms.append(grad_norm)

        opt.step()
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_weight_average:
            ewa.step(not_compiled_model, distributed_backend.is_master_process())

        # Synchronize GPU before stopping timer
        if "cuda" in cfg.device:
            torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t_start) / 1e9  # Convert to seconds
        
        # Accumulate training time (in seconds)
        cumulative_train_time += dt

        curr_iter += 1

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            train_perplexity = 2.71828**train_loss
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )

            if cfg.wandb:
                wandb_logs = {
                    "tokens": tokens,
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": train_perplexity,
                    "lr": current_lrs[0],
                    "iter_dt": dt,
                    "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
                    "mean_grad_norm": (
                        torch.tensor(grad_norms).mean().item() if grad_norms else 0
                    ),
                    # Log train loss vs training time (seconds)
                    "loss_vs_time/train_time_seconds": cumulative_train_time,
                    "loss_vs_time/train_loss": train_loss,
                    **train_aux_losses,
                }

                if cfg.log_parameter_norms:
                    raw_model = distributed_backend.get_raw_model(model)
                    model_norm = get_parameter_norms(raw_model, order=cfg.norm_order)
                    wandb_logs["model_norm"] = model_norm

                wandb.log(wandb_logs)

            grad_norms = []

    # Log final training statistics
    if distributed_backend.is_master_process() and cfg.wandb:
        total_training_time = time.time() - training_start_time
        wandb.log({
            "final/total_wall_time_seconds": total_training_time,
            "final/cumulative_train_time_seconds": cumulative_train_time,
            "final/avg_iter_time_seconds": cumulative_train_time / cfg.iterations if cfg.iterations > 0 else 0,
        })

    return stats


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
    cumulative_train_time=0.0,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    val_reader.set_step(0)

    # Synchronize GPU before starting evaluation timer
    if "cuda" in cfg.device:
        torch.cuda.synchronize()
    eval_start_time = time.perf_counter()

    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=(
            val_reader.num_batches()
            if curr_iter == cfg.iterations or full_eval
            else cfg.eval_batches
        ),
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    # Synchronize GPU before stopping evaluation timer
    if "cuda" in cfg.device:
        torch.cuda.synchronize()
    eval_time = time.perf_counter() - eval_start_time

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        # Always use val/* as key, and additionally log final-val/* at the last iteration
        logs = {
            "tokens": tokens,
            "iter": curr_iter,
            "val/loss": val_loss,
            "val/perplexity": val_perplexity,
            "val/acc": val_acc,
            # Log val loss vs training time (seconds)
            "loss_vs_time/train_time_seconds": cumulative_train_time,
            "loss_vs_time/val_loss": val_loss,
            **val_aux_losses,
        }
        
        # If at the last iteration or full_eval, additionally add final-val/* records
        if curr_iter == cfg.iterations or full_eval:
            logs.update({
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            })
        
        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()