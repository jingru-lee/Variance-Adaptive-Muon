wandb_log = True
wandb_project = 'GPT2-small-ICGAI'
wandb_run_name='muon-nsr'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

# GPT small config
n_layer = 12
n_head = 12
n_embd = 768

max_iters = 100000

# optimizer
optimizer_name = 'muon_nsr'
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

min_lr = 6e-5 

data_path = "/root/autodl-tmp/AdaMuon/gpt2/data"
dataset = "openwebtext-bin"

out_dir_name = 'small_muon_nsr'
