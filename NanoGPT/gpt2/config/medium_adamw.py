wandb_log = True
wandb_project = 'GPT2-medium-ICGAI'
wandb_run_name='adamw'

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 15

# GPT medium config
n_layer = 24
n_head = 16
n_embd = 1024

max_iters = 100000

# optimizer
optimizer_name = 'adam'
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

min_lr = 6e-5 

data_path = "/root/autodl-tmp/AdaMuon/gpt2/data"
dataset = "openwebtext-bin"

out_dir_name = 'medium_adamw'
