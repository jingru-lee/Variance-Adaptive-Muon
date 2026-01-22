import argparse
import copy
import inspect
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

import config
import distributed
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.base import train
from optim.muon import CombinedScheduler, DistributedMuon
from optim.muon_nsr import DistributedMuon_NSR
from optim.muon_vs import DistributedMuon_VS
from optim.schedule import cos_inf_schedule, wsd_schedule
from optim.soap import SOAP


def get_args():
    """Parse command-line arguments for the training experiment."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    final_args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )

    return final_args, parser


def main(args, parser):
    """
    Main training loop orchestrator.
    
    This function handles:
    - Distributed backend initialization and configuration
    - Model creation and distribution
    - Optimizer selection and instantiation
    - Learning rate scheduler setup
    - Training execution and metric tracking
    
    Args:
        args: Parsed command-line arguments containing model, data, optimizer configs
        parser: ArgumentParser object for accessing default values
    """
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    # Set up reproducibility: seed all random number generators
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    exp_name = get_exp_name(args, parser, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    
    # Initialize Weights and Biases for experiment tracking (master process only)
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args),
            entity=args.wandb_entity,
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")
        
        # Define time-based metrics for analyzing convergence vs wall-clock time
        wandb.define_metric("loss_vs_time/train_time_seconds")
        wandb.define_metric("loss_vs_time/train_loss", step_metric="loss_vs_time/train_time_seconds")
        wandb.define_metric("loss_vs_time/val_loss", step_metric="loss_vs_time/train_time_seconds")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    # Load training and validation datasets
    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    # Initialize model and move to device
    model = get_model(args).to(args.device)
    print(f"\nModel:\n{model}")

    # Apply distributed training transformations (DDP, FSDP, etc.)
    model = distributed_backend.transform_model(model)

    # Extract parameter groups for optimization with per-group hyperparameters
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs(
        config=args
    )
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    
    # Log parameter statistics
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    nonemb_param_cnt = (
        params_cnt
        - distributed_backend.get_raw_model(model).lm_head.weight.numel()
        - distributed_backend.get_raw_model(model).transformer.wte.weight.numel()
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {
                "parameters": params_cnt,
                "optimized_parameters": optimized_params_cnt,
                "non_embedding_parameters": nonemb_param_cnt,
            }
        )

    args.world_size = distributed_backend.get_world_size()

    # Instantiate optimizer based on configuration
    if args.opt == "adamw":
        device_type = "cuda" if "cuda" in args.device else "cpu"
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    elif args.opt == "soap":
        opt = SOAP(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=args.merge_dims,
            precondition_1d=args.precondition_1d,
            normalize_grads=args.normalize_grads,
            data_format=args.soap_data_format,
            correct_bias=args.correct_bias,
        )
    elif args.opt == "d-muon":
        """
        Standard Distributed Muon optimizer with full-matrix orthogonalization.
        
        Strategy:
        - 2D parameters: Muon with momentum + orthogonalization
        - Other parameters: AdamW fallback
        """
        opt = DistributedMuon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "d-muon-nsr":
        """
        Distributed Muon with Noise Suppression via Regularization (NSR).
        
        Strategy:
        - 2D parameters: Muon with noise-suppressed preprocessing
        - Other parameters: AdamW fallback
        
        The NSR variant improves stability through variance-aware preprocessing:
        M_tilde = M_hat / sqrt(M_hat^2 + α*V)
        """
        opt = DistributedMuon_NSR(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            weight_decay=args.weight_decay,
            alpha=args.muon_nrs_alpha,
        )
    elif args.opt == "d-muon-vs":
        """
        Distributed Muon with Variance-Stabilized preprocessing (VS).
        
        Strategy:
        - 2D parameters: Muon with variance-stabilized preprocessing
        - Other parameters: AdamW fallback
        
        The VS variant emphasizes variance stabilization for improved robustness:
        M_tilde = M_hat / sqrt(V)
        
        This variant shares the same experimental configuration as d-muon and d-muon-nsr
        for fair comparison across momentum-based orthogonalization approaches.
        """
        opt = DistributedMuon_VS(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {args.opt}. "
            f"Supported optimizers are: adamw, soap, d-muon, d-muon-nsr, d-muon-vs"
        )
    print(f"\nOptimizer:\n{opt}")

    # Set up learning rate scheduler
    if args.scheduler != "none":
        assert (
            args.warmup_steps < args.iterations
        ), "Warmup steps must be < iterations."
        if args.scheduler in ["cos", "linear"]:
            # OneCycleLR with cosine or linear annealing
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[group.get("lr", args.lr) for group in group_specs],
                total_steps=args.iterations,
                pct_start=args.warmup_steps / args.iterations,
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=1e2,
                final_div_factor=args.final_div_factor,
            )
        elif args.scheduler == "cos_inf":
            # Cosine annealing with infinite tail
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "wsd":
            # Warmup-then-Linear-Decay schedule
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    # Synchronize all processes before directory operations
    if args.distributed_backend is not None:
        torch.distributed.barrier()

    # Master process manages experiment directory creation
    if distributed_backend.is_master_process():
        if exp_dir.exists():
            print(f"Removing existing experiment directory: {exp_dir}")
            # Robust directory removal with retry logic
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(exp_dir)
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt + 1}/{max_retries} removing directory due to: {e}")
                        time.sleep(0.5)
                    else:
                        # Fallback: manual file-by-file deletion
                        print(f"Failed to remove directory after {max_retries} attempts, trying manual cleanup...")
                        try:
                            for root, dirs, files in os.walk(exp_dir, topdown=False):
                                for name in files:
                                    try:
                                        os.remove(os.path.join(root, name))
                                    except OSError:
                                        pass
                                for name in dirs:
                                    try:
                                        os.rmdir(os.path.join(root, name))
                                    except OSError:
                                        pass
                            os.rmdir(exp_dir)
                        except OSError:
                            print(f"Warning: Could not fully remove {exp_dir}, continuing anyway...")
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Wait for master process to complete directory operations
    if args.distributed_backend is not None:
        torch.distributed.barrier()

    # Execute training loop
    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_data_readers(args, verbose=True):
    """
    Initialize data readers for training and validation datasets.
    
    Args:
        args: Configuration arguments containing dataset parameters.
        verbose: Whether to print dataset statistics.
    
    Returns:
        dict: Dictionary with 'train' and 'val' DataReader instances.
    """
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # Identical per rank for consistent evaluation
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


def get_exp_name(
    args,
    parser,
    distributed_backend,
    key_args=["model", "dataset", "opt"],
    ignore_args=[
        "eval_interval",
        "full_eval_at",
        "distributed_backend",
        "latest_ckpt_interval",
        "permanent_ckpt_interval",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "batch_size",
        "acc_steps",
        "results_base_folder",
        "run_prefix",
        "wandb_run_prefix",
        "seed",
        "device",
        "adema_beta3_warmup",
        "adema_alpha_warmup",
        "plot_router_logits",
        "weight_average",
        "wa_dtype",
        "wa_use_temp_dir",
        "wa_sweep_horizon",
        "exponential_weight_average",
        "moe",
        "world_size",
    ],
):
    """
    Generate experiment name from configuration parameters.
    
    The experiment name encodes key hyperparameters and non-default settings
    to uniquely identify the training run and facilitate result organization.
    
    Args:
        args: Parsed arguments object.
        parser: ArgumentParser for accessing default values.
        distributed_backend: Distributed training backend.
        key_args: Arguments to include in the name prefix.
        ignore_args: Arguments to exclude from the name suffix.
    
    Returns:
        str: Unique experiment identifier string.
    """
    # Extract default values from parser
    defaults = vars(parser.parse_args([]))

    # Build prefix with key model/data/optimizer info
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if key == "model":
                # Add qualifiers for special training modes
                if getattr(args, "moe", False):
                    value = f"moe_{value}"
                if getattr(args, "weight_average", False):
                    value = f"{value}_WA"
                if getattr(args, "exponential_weight_average", False):
                    value = f"{value}_EWA"
            prefix_parts.append(f"{key}-{value}")

    prefix = "_".join(prefix_parts)
    prefix = f"{args.batch_size}x{args.acc_steps}_" + prefix

    # Build suffix with non-default hyperparameters
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args:
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            non_default_parts.append(f"{key}-{value}")

    non_default_string = "_".join(non_default_parts)

    if args.run_prefix is not None:
        prefix = args.run_prefix + "_" + prefix

    # Combine prefix and suffix
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


if __name__ == "__main__":
    args, parser = get_args()
    main(args, parser)