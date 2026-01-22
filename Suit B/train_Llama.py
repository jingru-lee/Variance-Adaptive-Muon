"""
Training script for Llama models with Muon-family optimizers.

This script supports distributed training with the following optimizers:
- Muon: Momentum Orthogonalized by Newton-Schulz
- Muon_NSR: Muon with Noise-to-Signal Ratio preprocessing
- Muon_VS: Muon with Variance-Stabilized preprocessing

Reference:
    - Muon: https://github.com/KellerJordan/Muon
    - Moonlight: https://github.com/MoonshotAI/Moonlight
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

import datasets
import numpy as np
import torch
import torch.distributed as dist
import transformers
import wandb
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.modeling_llama import LlamaForCausalLM

from muon import Muon
from muon_nsr import Muon_NSR
from muon_vs import Muon_VS

# Set transformer logging level to reduce verbosity
transformers.logging.set_verbosity_error()

# Filter out gradient checkpointing warnings
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*", category=UserWarning)


def parse_args(args=None):
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training hyperparameters.
    """
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to model configuration JSON file")
    parser.add_argument("--use_hf_model", default=False, action="store_true",
                        help="Use HuggingFace model implementation instead of custom")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to checkpoint directory for resuming training")
    
    # Batch size configuration
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=None,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--total_batch_size", type=int, default=None,
                        help="Total batch size across all devices and accumulation steps")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for training")
    
    # Optimizer configuration
    parser.add_argument("--optimizer", default="muon",
                        choices=["muon", "muon_nsr", "muon_vs"],
                        help="Optimizer type: muon, muon_nsr, or muon_vs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate for Muon parameters")
    parser.add_argument("--adamw_lr_multiplier", type=float, default=3.0,
                        help="Learning rate multiplier for AdamW parameters in Muon optimizer")
    
    # Learning rate schedule configuration
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["linear", "cosine", "cosine_restarts"],
                        help="Learning rate decay schedule type")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum learning rate as ratio of max learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1_000,
                        help="Number of warmup steps")
    parser.add_argument("--stable_steps", type=int, default=2_000,
                        help="Number of steps to keep learning rate constant after warmup")
    
    # Training configuration
    parser.add_argument("--activation_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay coefficient for L2 regularization")
    parser.add_argument("--eval_every", type=int, default=5_000,
                        help="Evaluation interval in update steps")
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Total number of update steps for training")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Maximum training tokens (overwrites num_training_steps). "
                             "Supports M and B suffixes, e.g., 100M or 1B")
    
    # Model saving configuration
    parser.add_argument("--save_every", type=int, default=10_000,
                        help="(Deprecated) Previously used for periodic saving")
    parser.add_argument("--save_dir", type=str, default="./pretrained_model",
                        help="Directory to save the final pretrained model")
    parser.add_argument("--no_save_model", action="store_false", dest="save_model",
                        help="Disable model saving after training")
    parser.set_defaults(save_model=True)
    
    # Experiment tracking
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for wandb logging")
    parser.add_argument("--name", type=str, default="test",
                        help="Experiment name for logging")
    parser.add_argument("--wandb_project", type=str, default="llama-130m-llama2-tokenizer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Weights & Biases run name (defaults to --name)")
    
    # System configuration
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                        help="Data type for model parameters (bfloat16 or float32)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--single_gpu", default=False, action="store_true",
                        help="Disable distributed training for single GPU")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging mode")
    
    # Gradient clipping configuration
    parser.add_argument("--grad_clipping", type=float, default=0.0,
                        help="Gradient clipping threshold (g_norm) for all optimizers")
    parser.add_argument("--g_norm", type=float, default=1.0,
                        help="Gradient norm threshold (deprecated, use --grad_clipping)")

    # AdamW betas for internal AdamW in Muon optimizers
    parser.add_argument('--betas', nargs='+', default=[0.9, 0.999], type=float,
                        help="Beta coefficients for AdamW momentum estimation")

    # Dataset configuration
    parser.add_argument("--dataset_info", type=str, required=True,
                        help="Path to dataset info JSON file")
    parser.add_argument("--cache_dir", type=str, default="autodl-tmp",
                        help="Data cache directory relative to current working directory")

    # Muon-family optimizer shared parameters
    parser.add_argument("--momentum", type=float, default=0.95,
                        help="Momentum coefficient for Muon/Muon_NSR/Muon_VS optimizers")
    parser.add_argument("--muon_eps", type=float, default=1e-8,
                        help="Epsilon for numerical stability in all Muon-family optimizers")
    
    # Muon_NSR specific parameters
    parser.add_argument("--muon_nsr_alpha", type=float, default=1000.0,
                        help="Alpha parameter for variance scaling in Muon_NSR optimizer")
    
    # Muon_VS specific parameters
    parser.add_argument("--muon_vs_alpha", type=float, default=1000.0,
                        help="Alpha parameter for variance scaling in Muon_VS optimizer")

    args = parser.parse_args(args)

    # Set wandb run name to experiment name if not specified
    if args.wandb_name is None:
        args.wandb_name = args.name

    # Use grad_clipping as g_norm if specified
    if args.grad_clipping > 0.0:
        args.g_norm = args.grad_clipping

    args = args_utils.check_args_torchrun_main(args)
    return args


def get_three_stage_scheduler(optimizer, warmup_steps, stable_steps, num_training_steps, 
                              min_lr_ratio=0.1, scheduler_type="cosine"):
    """
    Create a three-stage learning rate scheduler.
    
    The schedule consists of:
        1. Warmup: Linear increase from 0 to max_lr
        2. Stable: Constant learning rate at max_lr
        3. Decay: Linear or cosine decay from max_lr to min_lr
    
    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        stable_steps: Number of steps to maintain constant learning rate.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as ratio of maximum.
        scheduler_type: Type of decay ("linear" or "cosine").
    
    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured learning rate scheduler.
    """
    def lr_lambda(current_step):
        # Stage 1: Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Stage 2: Stable (constant learning rate)
        elif current_step < warmup_steps + stable_steps:
            return 1.0
        # Stage 3: Decay
        else:
            decay_steps = num_training_steps - warmup_steps - stable_steps
            current_decay_step = current_step - warmup_steps - stable_steps
            
            if decay_steps <= 0:
                return min_lr_ratio
            
            progress = float(current_decay_step) / float(decay_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            
            if scheduler_type == "linear":
                return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
            else:  # cosine
                import math
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sync_adamw_lr_with_scheduler(optimizer, adamw_lr_multiplier):
    """
    Synchronize adamw_lr with the current lr after scheduler update.
    
    This function is necessary because LambdaLR only updates the 'lr' key 
    in param_groups, but Muon-family optimizers use a separate 'adamw_lr' 
    key for AdamW parameters (embeddings, 1D params, lm_head).
    
    Args:
        optimizer: The Muon, Muon_NSR, or Muon_VS optimizer.
        adamw_lr_multiplier: Multiplier ratio (adamw_lr = lr * multiplier).
    """
    for group in optimizer.param_groups:
        if "adamw_lr" in group:
            # Get base lr from the group
            base_lr = group.get("lr", group.get("initial_lr", 1e-4))
            group["adamw_lr"] = base_lr * adamw_lr_multiplier


def classify_parameters_for_optimizer(model, optimizer_name="muon"):
    """
    Classify model parameters into groups for Muon-family optimizers.
    
    For Muon/Muon_NSR/Muon_VS, parameters are separated into two groups:
        - muon_params: 2D parameters (excluding embeddings and lm_head)
          These are optimized with full-matrix orthogonalization.
        - adamw_params: Embedding parameters, 1D parameters, and lm_head
          These are optimized with internal AdamW.

    Args:
        model: The model whose parameters are to be classified.
        optimizer_name: Name of the optimizer ('muon', 'muon_nsr', 'muon_vs').

    Returns:
        tuple: (muon_params, adamw_params) lists of parameters.
    """
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model

    # Handle torch.compile wrapped models
    if hasattr(actual_model, "_orig_mod"):
        actual_model = actual_model._orig_mod

    muon_params = []     # 2D parameters for full-matrix Muon optimization
    adamw_params = []    # Embedding, 1D, and lm_head parameters for AdamW
    
    # Classify embedding parameters into adamw_params
    if hasattr(actual_model, "model") and hasattr(actual_model.model, "embed_tokens"):
        for param in actual_model.model.embed_tokens.parameters():
            if param.requires_grad:
                adamw_params.append(param)
    
    # Classify lm_head parameters into adamw_params
    if hasattr(actual_model, "lm_head"):
        for param in actual_model.lm_head.parameters():
            if param.requires_grad:
                adamw_params.append(param)
    
    # Classify remaining parameters by dimensionality
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Skip already classified parameters
        if any(p is param for p in adamw_params):
            continue
            
        # 1D parameters use AdamW
        if param.dim() < 2:
            adamw_params.append(param)
        else:
            muon_params.append(param)
    
    return muon_params, adamw_params


class PreprocessedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for preprocessed datasets.
    
    Provides a simple interface for accessing preprocessed data samples.
    
    Args:
        dataset: The underlying dataset object.
        batch_size: Batch size for logging purposes.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Log dataset format information
        if len(dataset) > 0:
            logger.info(f"Dataset sample keys: {list(dataset[0].keys())}")
        else:
            logger.warning("Dataset is empty!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Flag to track if batch information has been logged
_sample_keys_logged = False


def collate_fn(batch):
    """
    Collate function for dataloader that handles tensor conversion.
    
    Ensures all batch data is properly converted to tensors with
    appropriate data types for model input.
    
    Args:
        batch: List of sample dictionaries from the dataset.
        
    Returns:
        dict: Batched tensors ready for model input.
    """
    global _sample_keys_logged

    batch_dict = {}

    # Log sample keys only once to avoid log spam
    if not _sample_keys_logged and len(batch) > 0:
        logger.info(f"Sample keys in batch: {list(batch[0].keys())}")
        _sample_keys_logged = True

    for key in batch[0].keys():
        try:
            values = [example[key] for example in batch]
            if isinstance(values[0], (list, np.ndarray)):
                batch_dict[key] = torch.tensor(np.array(values), dtype=torch.long)
            elif isinstance(values[0], torch.Tensor):
                batch_dict[key] = torch.stack(values)
            else:
                batch_dict[key] = torch.tensor(values)
        except Exception as e:
            logger.warning(f"Failed to collate key '{key}': {e}")

    # Ensure input_ids exists in batch
    if "input_ids" not in batch_dict:
        logger.warning("input_ids not found in batch. Creating default tensor.")
        batch_dict["input_ids"] = torch.zeros((len(batch), 256), dtype=torch.long)

    # Create attention_mask if missing
    if "attention_mask" not in batch_dict and "input_ids" in batch_dict:
        logger.warning("attention_mask not found. Creating from input_ids.")
        batch_dict["attention_mask"] = (batch_dict["input_ids"] != 0).long()

    return batch_dict


@torch.no_grad()
def evaluate_model(model, val_data, pad_idx, global_rank, world_size, device, batch_size):
    """
    Run evaluation on the validation dataset.
    """
    torch.cuda.synchronize()
    _time = time.time()

    val_dataset = PreprocessedDataset(val_data, batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 0
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_loader:
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                processed_batch[k] = v.to(device)

        if "input_ids" not in processed_batch:
            logger.error("No input_ids in evaluation batch!")
            continue

        labels = processed_batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**processed_batch, labels=labels).loss
        total_loss += loss.detach()

        batch_tokens = (processed_batch["input_ids"] != pad_idx).sum().item()
        # ✅ 修复：乘以 world_size，与 train_Llama.py 保持一致
        evaluated_on_tokens += batch_tokens * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    torch.cuda.synchronize()

    return total_loss, evaluated_on_tokens


def create_default_generation_config():
    """
    Create a default generation configuration.
    
    This avoids downloading configuration from HuggingFace Hub
    when saving models locally.
    
    Returns:
        GenerationConfig: Default generation configuration.
    """
    config = GenerationConfig()
    config.max_length = 256
    config.temperature = 1.0
    config.top_p = 0.9
    config.top_k = 50
    config.num_beams = 1
    config.do_sample = True
    return config


def unwrap_model(model):
    """
    Unwrap model from DDP and torch.compile wrappers.
    
    This is necessary to get the actual model for saving, as both
    DistributedDataParallel and torch.compile wrap the model.
    
    Args:
        model: The potentially wrapped model.
        
    Returns:
        The unwrapped model suitable for saving.
    """
    # Unwrap DDP
    if hasattr(model, "module"):
        model = model.module
    
    # Unwrap torch.compile
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    
    return model


def save_model_for_eval_harness(model, tokenizer, save_dir, training_state=None, 
                                 optimizer=None, scheduler=None, run_config=None, args=None):
    """
    Save model in a format compatible with lm-evaluation-harness.
    
    This function saves the model using HuggingFace's save_pretrained method,
    which creates all necessary files for loading with AutoModelForCausalLM.
    
    The saved format includes:
        - model.safetensors (or pytorch_model.bin as fallback)
        - config.json
        - generation_config.json
        - tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
        - training_state.json (optional)
        - optimizer.pt (optional)
    
    Args:
        model: The model to save (can be wrapped in DDP/torch.compile).
        tokenizer: The tokenizer to save.
        save_dir: Directory to save the model.
        training_state: Optional dict containing training state info.
        optimizer: Optional optimizer to save state.
        scheduler: Optional scheduler to save state.
        run_config: Optional run configuration dict.
        args: Optional parsed arguments.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model for eval-harness compatibility to {save_dir}")
    
    # Unwrap model from DDP and torch.compile
    model_to_save = unwrap_model(model)
    
    # Ensure model has generation_config
    if not hasattr(model_to_save, "generation_config") or model_to_save.generation_config is None:
        model_to_save.generation_config = create_default_generation_config()
        logger.info("Created default generation config")
    
    # Try to save with safetensors first (preferred by eval-harness)
    try:
        model_to_save.save_pretrained(
            save_dir,
            safe_serialization=True,  # Save as safetensors
            max_shard_size="10GB"  # Avoid sharding for smaller models
        )
        logger.info("Model saved with safetensors format (model.safetensors)")
    except Exception as e:
        logger.warning(f"Failed to save with safetensors: {e}")
        logger.info("Falling back to pytorch_model.bin format")
        try:
            model_to_save.save_pretrained(
                save_dir,
                safe_serialization=False,  # Save as pytorch_model.bin
                max_shard_size="10GB"
            )
            logger.info("Model saved with PyTorch format (pytorch_model.bin)")
        except Exception as e2:
            logger.error(f"Failed to save model with save_pretrained: {e2}")
            # Last resort: manual save
            logger.info("Attempting manual model save...")
            state_dict = model_to_save.state_dict()
            torch.save(state_dict, save_dir / "pytorch_model.bin")
            model_to_save.config.save_pretrained(save_dir)
            logger.info("Model saved manually")
    
    # Save tokenizer
    try:
        tokenizer.save_pretrained(save_dir)
        logger.info("Tokenizer saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save tokenizer: {e}")
    
    # Save generation config separately if not already saved
    gen_config_path = save_dir / "generation_config.json"
    if not gen_config_path.exists():
        try:
            model_to_save.generation_config.save_pretrained(save_dir)
            logger.info("Generation config saved separately")
        except Exception as e:
            logger.warning(f"Failed to save generation config: {e}")
    
    # Save training state
    if training_state is not None:
        training_state_path = save_dir / "training_state.json"
        with open(training_state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        logger.info(f"Training state saved to {training_state_path}")
    
    # Save optimizer state
    if optimizer is not None and scheduler is not None:
        optimizer_path = save_dir / "optimizer.pt"
        optimizer_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if run_config is not None:
            optimizer_state["config"] = run_config
        if args is not None:
            optimizer_state["dtype"] = args.dtype
        torch.save(optimizer_state, optimizer_path)
        logger.info(f"Optimizer state saved to {optimizer_path}")
    
    # Verify saved files
    expected_files = ["config.json"]
    optional_files = ["model.safetensors", "pytorch_model.bin", "generation_config.json", 
                      "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    
    logger.info("Verifying saved files:")
    for f in expected_files:
        file_path = save_dir / f
        if file_path.exists():
            logger.info(f"  ✓ {f} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            logger.error(f"  ✗ {f} (MISSING - required for eval-harness)")
    
    for f in optional_files:
        file_path = save_dir / f
        if file_path.exists():
            logger.info(f"  ✓ {f} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Check if model weights exist
    safetensors_path = save_dir / "model.safetensors"
    pytorch_path = save_dir / "pytorch_model.bin"
    
    if safetensors_path.exists():
        logger.info("Model weights saved as safetensors (preferred by eval-harness)")
    elif pytorch_path.exists():
        logger.info("Model weights saved as pytorch_model.bin")
    else:
        logger.error("No model weights found! Model saving may have failed.")
    
    logger.info(f"\nModel saved successfully to {save_dir}")
    logger.info("To evaluate with lm-evaluation-harness, run:")
    logger.info(f"  lm_eval --model hf --model_args pretrained={save_dir} --tasks <task_name>")


def main(args):
    """
    Main training function.
    
    Orchestrates the complete training pipeline including:
        - Distributed training setup
        - Data loading and preprocessing
        - Model initialization
        - Optimizer and scheduler configuration
        - Training loop with evaluation
        - Model checkpointing
    
    Args:
        args: Parsed command line arguments.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize distributed training environment
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    # Calculate gradient accumulation steps
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % (args.batch_size * world_size) == 0, \
                "total_batch_size must be divisible by batch_size * world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must equal total_batch_size"

    # Disable logging for non-master ranks
    if global_rank != 0:
        logger.remove()

    # Initialize wandb for experiment tracking
    if global_rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.tags.split(",") if args.tags else None
        )

        # Define metrics for various x-axes
        wandb.define_metric("global_step")
        wandb.define_metric("update_step")
        wandb.define_metric("tokens_seen")
        wandb.define_metric("training_time")

        wandb.define_metric("train_loss", step_metric="update_step")
        wandb.define_metric("eval_loss", step_metric="update_step")
        wandb.define_metric("lr", step_metric="update_step")
        wandb.define_metric("adamw_lr", step_metric="update_step")

        wandb.define_metric("train_loss_vs_time", step_metric="training_time")
        wandb.define_metric("eval_loss_vs_time", step_metric="training_time")

        wandb.define_metric("train_loss_vs_tokens", step_metric="tokens_seen")
        wandb.define_metric("eval_loss_vs_tokens", step_metric="tokens_seen")

    logger.info(f"Using distributed training with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the following arguments:")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # Determine decay type for logging
    decay_type = "linear" if args.scheduler == "linear" else "cosine"

    # Log learning rate schedule information
    logger.info("=" * 40)
    logger.info("Learning Rate Schedule (Three-Stage):")
    logger.info(f"  Stage 1 - Warmup:   steps 0 to {args.warmup_steps}")
    logger.info(f"  Stage 2 - Stable:   steps {args.warmup_steps} to {args.warmup_steps + args.stable_steps}")
    logger.info(f"  Stage 3 - Decay:    steps {args.warmup_steps + args.stable_steps} to {args.num_training_steps} ({decay_type} decay)")
    logger.info(f"  Max LR:             {args.lr}")
    logger.info(f"  Min LR:             {args.lr * args.min_lr_ratio}")
    logger.info(f"  AdamW LR Multiplier: {args.adamw_lr_multiplier}")
    logger.info(f"  Max AdamW LR:       {args.lr * args.adamw_lr_multiplier}")
    logger.info(f"  Min AdamW LR:       {args.lr * args.min_lr_ratio * args.adamw_lr_multiplier}")
    logger.info(f"  Momentum:           {args.momentum}")
    logger.info(f"  Gradient Norm (g_norm): {args.g_norm}")
    logger.info(f"  Muon Eps:           {args.muon_eps}")
    if args.optimizer.lower() == "muon_nsr":
        logger.info(f"  Muon_NSR Alpha:     {args.muon_nsr_alpha}")
    if args.optimizer.lower() == "muon_vs":
        logger.info(f"  Muon_VS Alpha:      {args.muon_vs_alpha}")
    logger.info("=" * 40)

    # Create data cache directory
    cache_dir = os.path.join(os.getcwd(), args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Using cache directory: {cache_dir}")

    # Load dataset info
    logger.info(f"Reading dataset info from: {args.dataset_info}")
    with open(args.dataset_info, "r") as f:
        dataset_info = json.load(f)

    train_path = dataset_info["train_path"]
    eval_path = dataset_info["eval_path"]

    logger.info(f"Training data path: {train_path}")
    logger.info(f"Validation data path: {eval_path}")
    logger.info(f"Number of training samples: {dataset_info.get('num_train_samples', 'N/A')}")
    logger.info(f"Number of validation samples: {dataset_info.get('num_eval_samples', 'N/A')}")

    # Load tokenizer
    tokenizer_name = dataset_info.get("tokenizer", "autodl-tmp/tokenizers/llama-2-7b-hf")
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            model_max_length=args.max_length,
            cache_dir=cache_dir,
            use_fast=True
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer {tokenizer_name}: {str(e)}")
        logger.error("Please ensure HF_TOKEN is set and you have access to the model")
        sys.exit(1)
    
    # Set pad_token for Llama 2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        logger.info(f"Set pad_token to unk_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer bos_token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    logger.info(f"Tokenizer eos_token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    logger.info(f"Tokenizer pad_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    logger.info(f"Tokenizer unk_token: '{tokenizer.unk_token}' (id={tokenizer.unk_token_id})")
    
    # Verify tokenizer vocab size
    if tokenizer.vocab_size != 32000:
        logger.warning(f"WARNING: Expected vocab_size=32000 for Llama 2, got {tokenizer.vocab_size}")
    else:
        logger.info(f"Tokenizer vocab_size matches Llama 2 (32000)")

    # Load preprocessed training data
    logger.info(f"Loading preprocessed training data: {train_path}")
    train_data = datasets.Dataset.load_from_disk(train_path)

    if len(train_data) > 0:
        logger.info(f"Training dataset features: {train_data.features}")
        logger.info(f"First training sample keys: {list(train_data[0].keys())}")
        sample = train_data[0]
        for k, v in sample.items():
            if isinstance(v, (list, np.ndarray)):
                logger.info(f"  {k}: shape={len(v)}, dtype={type(v[0]) if len(v) > 0 else 'N/A'}")
            else:
                logger.info(f"  {k}: {type(v)}")

    # Load validation data
    logger.info(f"Loading preprocessed validation data: {eval_path}")
    eval_data = datasets.Dataset.load_from_disk(eval_path)

    if len(eval_data) > 0:
        logger.info(f"First validation sample keys: {list(eval_data[0].keys())}")

    # Create data loader with distributed sampler
    train_dataset = PreprocessedDataset(train_data, args.batch_size)

    if not args.single_gpu and world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=args.seed
        )
        logger.info(f"Using DistributedSampler for data sharding on rank {global_rank}/{world_size}")
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
        logger.info("Using normal DataLoader, shuffle=True")

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    # Initialize model
    logger.info(f"Loading model config from: {args.model_config}")
    model_config = AutoConfig.from_pretrained(args.model_config)
    
    logger.info(f"Model config vocab_size: {model_config.vocab_size}")
    logger.info(f"Model config bos_token_id: {model_config.bos_token_id}")
    logger.info(f"Model config eos_token_id: {model_config.eos_token_id}")
    logger.info(f"Model config pad_token_id: {model_config.pad_token_id}")
    
    # Update token IDs to match tokenizer
    if model_config.vocab_size == 32000:
        logger.info("Updating model config token IDs to match Llama 2 tokenizer...")
        model_config.bos_token_id = tokenizer.bos_token_id
        model_config.eos_token_id = tokenizer.eos_token_id
        model_config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Updated bos_token_id: {model_config.bos_token_id}")
        logger.info(f"Updated eos_token_id: {model_config.eos_token_id}")
        logger.info(f"Updated pad_token_id: {model_config.pad_token_id}")
    else:
        logger.warning(f"Model vocab_size ({model_config.vocab_size}) does not match expected 32000")
    
    # Verify vocab size compatibility
    if model_config.vocab_size != tokenizer.vocab_size:
        logger.error(f"MISMATCH: Model vocab_size ({model_config.vocab_size}) != Tokenizer vocab_size ({tokenizer.vocab_size})")
        logger.error("Please ensure your model config and tokenizer are compatible!")
        sys.exit(1)
    else:
        logger.info(f"Model and tokenizer vocab_size match: {model_config.vocab_size}")
    
    # Disable cache for gradient checkpointing
    if args.activation_checkpointing:
        model_config.use_cache = False

    # Initialize model
    if args.use_hf_model:
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    # Enable gradient checkpointing
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Compile model with torch.compile
    if torch.__version__ >= "2.0.0":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Initialize training state
    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    total_training_time = 0.0

    # Resume from checkpoint if specified
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        safetensors_path = os.path.join(args.continue_from, "model.safetensors")
        
        # Try safetensors first, then fall back to pytorch_model.bin
        if os.path.exists(safetensors_path):
            logger.info(f"Loading from safetensors: {safetensors_path}")
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
                unwrapped = unwrap_model(model)
                unwrapped.load_state_dict(state_dict, strict=True)
            except ImportError:
                logger.warning("safetensors not installed, falling back to pytorch_model.bin")
                if os.path.exists(checkpoint_path):
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    unwrapped = unwrap_model(model)
                    unwrapped.load_state_dict(state_dict, strict=True)
        elif os.path.exists(checkpoint_path):
            logger.info(f"Loading from pytorch_model.bin: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            unwrapped = unwrap_model(model)
            unwrapped.load_state_dict(state_dict, strict=True)
        else:
            logger.error(f"No model weights found in {args.continue_from}")
            sys.exit(1)
            
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen"]
            if "total_training_time" in _old_state:
                total_training_time = _old_state["total_training_time"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            beginning_step = update_step
        else:
            logger.warning("No training_state.json found, starting from scratch")
        logger.info("*" * 40)

    # Move model to device and set precision
    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    # Count model parameters
    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Initialize wandb config
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'dclm_baseline',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
        "tokenizer_name": tokenizer_name,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_vocab_size": model_config.vocab_size,
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        wandb.save(script_path, base_path=script_dir, policy="now")
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    # Log model information
    logger.info(f"\n{model}\n")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    if args.save_model:
        logger.info(f"Model will be saved to {args.save_dir} after training completes")
    else:
        logger.info(f"Model saving is disabled (--no_save_model flag is set)")

    # Initialize optimizer
    optimizer_name = args.optimizer.lower()
    
    # Classify parameters for Muon-family optimizers
    logger.info(f"Using {optimizer_name.upper()} optimizer with two-group parameter classification")
    muon_params, adamw_params = classify_parameters_for_optimizer(model, optimizer_name=optimizer_name)
    
    logger.info(f"  - Muon optimization (2D non-embedding, non-lm_head): {len(muon_params)} tensors")
    logger.info(f"  - AdamW optimization (embed + 1D + lm_head): {len(adamw_params)} tensors")
    
    # Calculate actual learning rates with multipliers
    adamw_lr_actual = args.lr * args.adamw_lr_multiplier
    
    if optimizer_name == "muon":
        optimizer = Muon(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            wd=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
            ns_steps=5,
            adamw_lr=adamw_lr_actual,
            adamw_wd=args.weight_decay,
            adamw_betas=tuple(args.betas),
            adamw_eps=args.muon_eps,
            g_norm=args.g_norm,
        )
        
        logger.info(f"Muon config:")
        logger.info(f"  - Muon lr: {args.lr}, wd: {args.weight_decay}, momentum: {args.momentum}, ns_steps: 5")
        logger.info(f"  - Muon eps: {args.muon_eps}")
        logger.info(f"  - AdamW lr: {args.lr} * {args.adamw_lr_multiplier} = {adamw_lr_actual}, wd: {args.weight_decay}")
        logger.info(f"  - Gradient clipping g_norm: {args.g_norm}")
        logger.info(f"  - AdamW lr will be synced with scheduler (adamw_lr = lr * {args.adamw_lr_multiplier})")
        logger.info("Optimization strategy:")
        logger.info("  - Group 1: 2D params (non-embed, non-lm_head) -> full-matrix Newton-Schulz (Muon)")
        logger.info("  - Group 2: Embeddings + 1D + lm_head -> AdamW with adamw_lr")
        
    elif optimizer_name == "muon_nsr":
        optimizer = Muon_NSR(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            wd=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
            ns_steps=5,
            adamw_lr=adamw_lr_actual,
            adamw_wd=args.weight_decay,
            adamw_betas=tuple(args.betas),
            adamw_eps=args.muon_eps,
            alpha=args.muon_nsr_alpha,
            g_norm=args.g_norm,
        )
        
        logger.info(f"Muon_NSR config:")
        logger.info(f"  - Muon_NSR lr: {args.lr}, wd: {args.weight_decay}, momentum: {args.momentum}, ns_steps: 5")
        logger.info(f"  - Muon_NSR alpha: {args.muon_nsr_alpha}, eps: {args.muon_eps}")
        logger.info(f"  - AdamW lr: {args.lr} * {args.adamw_lr_multiplier} = {adamw_lr_actual}, wd: {args.weight_decay}")
        logger.info(f"  - Gradient clipping g_norm: {args.g_norm}")
        logger.info(f"  - AdamW lr will be synced with scheduler (adamw_lr = lr * {args.adamw_lr_multiplier})")
        logger.info("Optimization strategy:")
        logger.info("  - Group 1: 2D params (non-embed, non-lm_head) -> Muon_NSR with variance-aware preprocessing")
        logger.info("  - Group 2: Embeddings + 1D + lm_head -> AdamW with adamw_lr")
        
    elif optimizer_name == "muon_vs":
        optimizer = Muon_VS(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            wd=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
            ns_steps=5,
            adamw_lr=adamw_lr_actual,
            adamw_wd=args.weight_decay,
            adamw_betas=tuple(args.betas),
            adamw_eps=args.muon_eps,
            alpha=args.muon_vs_alpha,
            g_norm=args.g_norm,
        )
        
        logger.info(f"Muon_VS config:")
        logger.info(f"  - Muon_VS lr: {args.lr}, wd: {args.weight_decay}, momentum: {args.momentum}, ns_steps: 5")
        logger.info(f"  - Muon_VS alpha: {args.muon_vs_alpha}, eps: {args.muon_eps}")
        logger.info(f"  - AdamW lr: {args.lr} * {args.adamw_lr_multiplier} = {adamw_lr_actual}, wd: {args.weight_decay}")
        logger.info(f"  - Gradient clipping g_norm: {args.g_norm}")
        logger.info(f"  - AdamW lr will be synced with scheduler (adamw_lr = lr * {args.adamw_lr_multiplier})")
        logger.info("Optimization strategy:")
        logger.info("  - Group 1: 2D params (non-embed, non-lm_head) -> Muon_VS with variance-scaled preprocessing")
        logger.info("  - Group 2: Embeddings + 1D + lm_head -> AdamW with adamw_lr")
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # Create learning rate scheduler
    logger.info(f"Creating three-stage learning rate scheduler with {args.scheduler} decay...")
    scheduler = get_three_stage_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        num_training_steps=args.num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
        scheduler_type=args.scheduler,
    )

    # Wrap model with DDP for multi-GPU training
    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # Setup training variables
    pad_idx = tokenizer.pad_token_id
    torch.cuda.synchronize()
    update_time = time.time()
    local_step = 0

    # Track last evaluation point
    last_eval_time = total_training_time
    last_eval_tokens = tokens_seen
    last_eval_step = update_step

    # ##############################
    # TRAINING LOOP
    # ##############################

    # Set sampler epoch for reproducibility
    if train_sampler is not None:
        fixed_epoch = args.seed * 13 + update_step * 17 + global_rank * 31
        train_sampler.set_epoch(fixed_epoch)
        logger.info(f"Set fixed sampler epoch to {fixed_epoch}")

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        # Check training termination condition
        if update_step >= args.num_training_steps:
            logger.info(f"Reached target training steps: {args.num_training_steps}")
            break

        # Process batch and move to device
        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                processed_batch[k] = v.to(device)

        if "input_ids" not in processed_batch:
            logger.error("No input_ids in batch!")
            continue

        # Create labels (shift is handled inside model)
        labels = processed_batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        # Forward pass
        outputs = model(**processed_batch, labels=labels)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        # Count tokens in this batch
        batch_tokens = (processed_batch["input_ids"] != pad_idx).sum().item()
        tokens_seen += batch_tokens * world_size

        # Update weights after accumulation
        if global_step % args.gradient_accumulation == 0:
            # Gradient clipping is handled inside Muon optimizers via g_norm
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Sync AdamW lr with scheduler
            sync_adamw_lr_with_scheduler(optimizer, args.adamw_lr_multiplier)

            update_step += 1

            # Get current learning rates
            current_lr = optimizer.param_groups[0].get("lr", args.lr)
            current_adamw_lr = optimizer.param_groups[0].get("adamw_lr", current_lr * args.adamw_lr_multiplier)

            # Calculate step time
            step_time = time.time() - update_time
            total_training_time += step_time

            # Logging
            if global_rank == 0:
                pbar.update(1)
                
                if update_step % 10 == 0:
                    tokens_per_second = batch_tokens * args.gradient_accumulation * world_size / step_time
                    logger.info(
                        f"Step {update_step}/{args.num_training_steps} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"AdamW_LR: {current_adamw_lr:.2e} | "
                        f"Tokens/s: {tokens_per_second:.0f} | "
                        f"Time: {total_training_time:.1f}s"
                    )

                wandb.log({
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "training_time": total_training_time,
                    "train_loss": loss.item(),
                    "lr": current_lr,
                    "adamw_lr": current_adamw_lr,
                    "train_loss_vs_time": loss.item(),
                    "train_loss_vs_tokens": loss.item(),
                }, step=update_step)

            update_time = time.time()

            # Run evaluation at specified intervals
            if update_step % args.eval_every == 0:
                eval_time_elapsed = total_training_time - last_eval_time
                eval_tokens_elapsed = tokens_seen - last_eval_tokens
                eval_steps_elapsed = update_step - last_eval_step

                logger.info(f"Running evaluation at step {update_step}, {eval_tokens_elapsed:,} tokens seen "
                            f"since last eval, {eval_time_elapsed:.2f}s elapsed")

                model.eval()
                eval_loss, eval_tokens = evaluate_model(
                    model, eval_data, pad_idx, global_rank, world_size, device, args.batch_size
                )
                model.train()

                if global_rank == 0:
                    logger.info(f"Eval loss: {eval_loss:.4f}")
                    wandb.log({
                        "eval_loss": eval_loss,
                        "eval_loss_vs_time": eval_loss,
                        "eval_loss_vs_tokens": eval_loss,
                        "eval_tokens": eval_tokens,
                    }, step=update_step)

                last_eval_time = total_training_time
                last_eval_tokens = tokens_seen
                last_eval_step = update_step

                # Reset update time after evaluation
                update_time = time.time()

    # ##############################
    # END OF TRAINING LOOP
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    # Save final model only after training completes (using eval-harness compatible format)
    if args.save_model and global_rank == 0:
        final_model_directory = args.save_dir
        logger.info(f"Saving final pretrained model to {final_model_directory}")
        
        # Prepare training state
        training_state = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "total_training_time": total_training_time,
        }
        
        # Save model using eval-harness compatible function
        save_model_for_eval_harness(
            model=model,
            tokenizer=tokenizer,
            save_dir=final_model_directory,
            training_state=training_state,
            optimizer=optimizer,
            scheduler=scheduler,
            run_config=run_config,
            args=args
        )
        
        logger.info(f"Final model saved successfully to {final_model_directory}")
        logger.info(f"  - Model weights: model.safetensors (or pytorch_model.bin)")
        logger.info(f"  - Model config: config.json")
        logger.info(f"  - Generation config: generation_config.json")
        logger.info(f"  - Tokenizer files: tokenizer.json, tokenizer_config.json, etc.")
        logger.info(f"  - Training state: training_state.json")
        logger.info(f"  - Optimizer state: optimizer.pt")
        logger.info("")
        logger.info("To evaluate with lm-evaluation-harness:")
        logger.info(f"  lm_eval --model hf --model_args pretrained={final_model_directory} --tasks <task_name>")

    # Cleanup
    if global_rank == 0:
        wandb.finish()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)