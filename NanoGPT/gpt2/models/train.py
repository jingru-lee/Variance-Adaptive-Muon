import os
import sys
import math
import time
import pickle
from contextlib import nullcontext
from runpy import run_path
from ast import literal_eval

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT


# =============================================================================
# I/O & RUN CONTROL
# =============================================================================
root_dir = "./"
data_path = root_dir + "/data"
out_dir_name = "out"                          
out_dir = root_dir + '/ckpts/' + out_dir_name
init_from = "scratch"                    # 'scratch' | 'resume' | 'gpt2*'
eval_only = False
always_save_checkpoint = False

# =============================================================================
# LOGGING
# =============================================================================
wandb_log = False
wandb_project = "gpt2"
wandb_run_name = "gpt2"
log_interval = 10
eval_interval = 1000
eval_iters = 200

# =============================================================================
# DATA & BATCHING
# =============================================================================
dataset = "openwebtext-bin"
block_size = 1024
batch_size = 12
gradient_accumulation_steps = 5

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Positional encoding and initialization configuration
use_rope = True
rope_base = 10000.0
init_method = "gpt2"
init_value = 0.006

# =============================================================================
# OPTIMIZATION
# =============================================================================
optimizer_name = "adamuon"               # 'adam' | 'muon' | 'adamuon' | 'muon_nsr' | 'muon_vs'
beta1 = 0.9
beta2 = 0.95

# =============================================================================
# TRAINING LENGTH
# =============================================================================
max_iters = 100000

# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================
warmup_iters = 2000
min_lr = 6e-5
learning_rate = 6e-4
weight_decay = 1e-1
grad_clip = 1.0

# =============================================================================
# SYSTEM / DISTRIBUTED / PRECISION
# =============================================================================
backend = "nccl"                         # DDP backend
device = "cuda"                          # 'cpu' | 'cuda[:id]' | 'mps'
dtype = "bfloat16"                       # 'float32' | 'bfloat16' | 'float16'
compile = True                           # Enable torch.compile for optimization

# =============================================================================
# Configuration Loading Utilities
# =============================================================================
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
        
config = {k: globals()[k] for k in config_keys}

# =============================================================================
# Distributed Data Parallel (DDP) Initialization
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1
ddp_world_size = int(os.environ['WORLD_SIZE']) if ddp else 1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    
# =============================================================================
# Random Seed and Matrix Multiplication Configuration
# =============================================================================
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# =============================================================================
# Data Loader (Memory-mapped files for efficient I/O)
# =============================================================================
data_dir = os.path.join(data_path, dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    """
    Fetch a batch of training or validation data.
    
    Args:
        split: 'train' or 'val' to specify which dataset to sample from.
        
    Returns:
        Tuple of (input_ids, target_ids) tensors on the specified device.
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # Pin memory for asynchronous GPU transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# =============================================================================
# Model Initialization
# =============================================================================
iter_num = 0
best_val_loss = 1e9

# Load vocabulary size from metadata if available
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_rope=use_rope, rope_base=rope_base,
                  init_method=init_method, init_value=init_value)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
               'use_rope', 'rope_base', 'init_method', 'init_value']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
               'use_rope', 'rope_base', 'init_method', 'init_value']:
        model_args[k] = getattr(model.config, k)

# Crop block size if the model was trained with a larger context
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# =============================================================================
# Automatic Mixed Precision (AMP) Scaler
# Only enabled for float16; bfloat16 does not require gradient scaling
# =============================================================================
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# =============================================================================
# Optimizer Configuration
# =============================================================================
params = list(model.parameters())
optimizer_list = model.configure_optimizers(
    optimizer_name, weight_decay, learning_rate, (beta1, beta2), device_type
)

# Resume optimizer states if loading from checkpoint
if init_from == 'resume':
    for idx, optimizer in enumerate(optimizer_list):
        if 'optimizers' in checkpoint and idx < len(checkpoint['optimizers']):
            optimizer.load_state_dict(checkpoint['optimizers'][idx])
    del state_dict
    del checkpoint

# =============================================================================
# Model Compilation (PyTorch 2.0+)
# =============================================================================
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# =============================================================================
# Distributed Data Parallel Wrapper
# =============================================================================
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
else:
    raw_model = model

# =============================================================================
# Evaluation Helper Function
# =============================================================================
@torch.no_grad()
def estimate_loss():
    """
    Estimate training and validation loss by averaging over multiple batches.
    
    Returns:
        Dictionary with 'train' and 'val' keys containing mean loss values.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =============================================================================
# Learning Rate Scheduler
# =============================================================================
def get_lr(it):
    """
    Compute learning rate using warmup-stable schedule.
    
    Args:
        it: Current iteration number.
        
    Returns:
        Learning rate for the current iteration.
    """
    # Linear warmup phase
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Stable phase (constant learning rate)
    else:
        return learning_rate

# =============================================================================
# Weights & Biases Initialization
# =============================================================================
if wandb_log and master_process:
    try:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        # Define custom x-axis metric for time vs loss visualization (time in seconds)
        wandb.define_metric("time_seconds")
        wandb.define_metric("train/loss_vs_time", step_metric="time_seconds")
        wandb.define_metric("val/loss_vs_time", step_metric="time_seconds")
    except Exception as e:
        print(f"[W&B] init failed: {e}")
        wandb_log = False

# =============================================================================
# Training Loop
# =============================================================================
X, Y = get_batch('train')  # Prefetch the first batch
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
clip_time = 0

# Cumulative training time in seconds (excludes evaluation overhead)
cumulative_train_time = 0.0

while True:
    # Update learning rate for current iteration
    lr = get_lr(iter_num)
    for opt in optimizer_list:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    # ==========================================================================
    # Evaluation and Checkpointing
    # ==========================================================================
    if iter_num % eval_interval == 0 and master_process:
        # Synchronize GPU to ensure accurate timing measurement
        if device_type == 'cuda':
            torch.cuda.synchronize()
        
        # Record cumulative training time up to this point (excluding evaluation)
        current_train_time = cumulative_train_time
        
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {current_train_time:.2f}s")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
                # Log time vs loss data (time in seconds)
                "time_seconds": current_train_time,
                "train/loss_vs_time": losses['train'],
                "val/loss_vs_time": losses['val'],
            }, step=iter_num)

        # Save checkpoint if validation loss improved
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    "optimizers": [opt.state_dict() for opt in optimizer_list],
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # Periodic checkpoint saving (every 5 evaluation intervals)
        if iter_num % (eval_interval * 5) == 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                "optimizers": [opt.state_dict() for opt in optimizer_list],
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            # torch.save(checkpoint, os.path.join(out_dir, 'ckpt_'+str(iter_num)+'.pt'))

    if iter_num == 0 and eval_only:
        break

    # Record iteration start time for accurate timing
    iter_start_time = time.time()

    # ==========================================================================
    # Forward/Backward Pass with Gradient Accumulation
    # ==========================================================================
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # Only synchronize gradients on the last micro-step for efficiency
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
        
        # Prefetch next batch asynchronously during backward pass
        X, Y = get_batch('train')
        
        # Backward pass with gradient scaling for mixed precision
        scaler.scale(loss).backward()

    # ==========================================================================
    # Gradient Clipping
    # ==========================================================================
    if grad_clip != 0.0:
        for opt in optimizer_list:
            scaler.unscale_(opt)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
    
    # ==========================================================================
    # Optimizer Step
    # ==========================================================================
    for opt in optimizer_list:
        scaler.step(opt)
    scaler.update()

    # Zero gradients for next iteration
    for opt in optimizer_list:
        opt.zero_grad(set_to_none=True)

    # Synchronize GPU to ensure accurate timing measurement
    if device_type == 'cuda':
        torch.cuda.synchronize()

    # ==========================================================================
    # Timing and Logging
    # ==========================================================================
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    # Accumulate training time (only training iterations, excluding evaluation)
    iter_time = t1 - iter_start_time
    cumulative_train_time += iter_time

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()
        if local_iter_num >= 5:
            try:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            except Exception:
                pass

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, cumulative_time {cumulative_train_time:.2f}s, mfu {running_mfu*100:.2f}%")

        if wandb_log:
            log_data = {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "train/clip_rate": clip_time / (iter_num + 1),
                "mfu": running_mfu * 100.0,
            }
            wandb.log(log_data, step=iter_num)

    iter_num += 1
    local_iter_num += 1

    # Termination condition
    if iter_num > max_iters:
        break

# =============================================================================
# Cleanup
# =============================================================================
if ddp:
    destroy_process_group()