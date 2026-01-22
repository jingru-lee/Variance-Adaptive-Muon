# 210M AdamW
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 24 \
    --batch_size 16 --sequence_length 512 --acc_steps 16 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.999 \
    --wandb --wandb_project llm_benchmark-210M-IGCAI \
    --eval_interval 1000 --latest_ckpt_interval 1000

# 210M D-Muon
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 24 \
    --batch_size 16 --sequence_length 512 --acc_steps 16 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt d-muon --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 \
    --momentum 0.95 --nesterov True \
    --wandb --wandb_project llm_benchmark-210M-IGCAI \
    --eval_interval 1000 --latest_ckpt_interval 1000

# 210M D-Muon NSR
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 24 \
    --batch_size 16 --sequence_length 512 --acc_steps 16 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt d-muon-nsr --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 \
    --momentum 0.95 --nesterov True \
    --wandb --wandb_project llm_benchmark-210M-IGCAI \
    --eval_interval 1000 --latest_ckpt_interval 1000

# 210M D-Muon VS
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 24 \
    --batch_size 16 --sequence_length 512 --acc_steps 16 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt d-muon-vs --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 \
    --momentum 0.95 --nesterov True \
    --wandb --wandb_project llm_benchmark-210M-IGCAI \
    --eval_interval 1000 --latest_ckpt_interval 1000

# 720M AdamW
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 2048 --n_head 16 --n_layer 12 \
    --batch_size 8 --sequence_length 512 --acc_steps 248 \
    --dataset fineweb --iterations 48000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.999 \
    --wandb --wandb_project llm_benchmark-720M-IGCAI \
    --eval_interval 500 --latest_ckpt_interval 1000

# 720M D-Muon
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 2048 --n_head 16 --n_layer 12 \
    --batch_size 8 --sequence_length 512 --acc_steps 248 \
    --dataset fineweb --iterations 48000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt d-muon --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.99 \
    --momentum 0.95 --nesterov True \
    --wandb --wandb_project llm_benchmark-720M-IGCAI \
    --eval_interval 500 --latest_ckpt_interval 1000

# 720M D-Muon NSR
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 2048 --n_head 16 --n_layer 12 \
    --batch_size 8 --sequence_length 512 --acc_steps 248 \
    --dataset fineweb --iterations 48000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt d-muon-nsr --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.99 \
    --momentum 0.95 --nesterov True \
    --wandb --wandb_project llm_benchmark-720M-IGCAI \
    --eval_interval 500 --latest_ckpt_interval 1000

# 720M D-Muon VS
torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 2048 --n_head 16 --n_layer 12 \
    --batch_size 8 --sequence_length 512 --acc_steps 248 \
    --dataset fineweb --iterations 48000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt d-muon-vs --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.99 \
    --momentum 0.95 --nesterov True \
    --wandb --wandb_project llm_benchmark-720M-IGCAI \
    --eval_interval 500 --latest_ckpt_interval 1000