# ============================================================================
# Training Scripts for Llama Models with Muon-Family Optimizers
# ============================================================================
# This script contains training configurations for Llama models of various
# sizes (130M, 300M, 1.2B) using Muon, Muon_NSR, and Muon_VS optimizers.
#
# Reference:
#   - Muon: https://github.com/KellerJordan/Muon
#   - Moonlight: https://github.com/MoonshotAI/Moonlight
# ============================================================================

# ============================================================================
# Llama 130M Experiments
# ============================================================================

# Muon optimizer for Llama 130M
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_130m.json \
    --lr 1.6e-2 \
    --adamw_lr_multiplier 0.2 \
    --batch_size 16 \
    --total_batch_size 128 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 5000 \
    --warmup_steps 0 \
    --stable_steps 1000 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 100 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon \
    --momentum 0.95 \
    --muon_eps 1e-15 \
    --save_dir checkpoints/llama_130m_4096seq \
    --name llama_130m_muon \
    --wandb_project "stf-igcai-130M" \
    --wandb_name "muon" \
    --dataset_info data/dataset_info_llama2.json

# Muon_NSR optimizer for Llama 130M
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_130m.json \
    --lr 1.6e-2 \
    --adamw_lr_multiplier 0.2 \
    --batch_size 16 \
    --total_batch_size 128 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 5000 \
    --warmup_steps 0 \
    --stable_steps 1000 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 100 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon_nsr \
    --momentum 0.95 \
    --muon_eps 1e-15 \
    --muon_nsr_alpha 1000.0 \
    --save_dir checkpoints/llama_130m_4096seq \
    --name llama_130m_muon_nsr \
    --wandb_project "stf-igcai-130M" \
    --wandb_name "muon_nsr-alpha=1000" \
    --dataset_info data/dataset_info_llama2.json

# Muon_VS optimizer for Llama 130M
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_130m.json \
    --lr 1.6e-2 \
    --adamw_lr_multiplier 0.2 \
    --batch_size 16 \
    --total_batch_size 128 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 5000 \
    --warmup_steps 0 \
    --stable_steps 1000 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 100 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon_vs \
    --momentum 0.95 \
    --muon_eps 1e-15 \
    --muon_vs_alpha 1000.0 \
    --save_dir checkpoints/llama_130m_4096seq \
    --name llama_130m_muon_vs \
    --wandb_project "stf-igcai-130M" \
    --wandb_name "muon_vs" \
    --dataset_info data/dataset_info_llama2.json


# ============================================================================
# Llama 300M Experiments
# ============================================================================

# Muon optimizer for Llama 300M
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_300m.json \
    --lr 8e-3 \
    --adamw_lr_multiplier 0.3 \
    --batch_size 16 \
    --total_batch_size 128 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 11500 \
    --warmup_steps 0 \
    --stable_steps 2300 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 100 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon \
    --momentum 0.98 \
    --muon_eps 1e-15 \
    --save_dir checkpoints/llama_300m_4096seq \
    --name llama_300m_muon \
    --wandb_project "stf-igcai-300M" \
    --wandb_name "muon" \
    --dataset_info data/dataset_info_llama2.json

# Muon_NSR optimizer for Llama 300M
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_300m.json \
    --lr 8e-3 \
    --adamw_lr_multiplier 0.3 \
    --batch_size 16 \
    --total_batch_size 128 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 11500 \
    --warmup_steps 0 \
    --stable_steps 2300 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 100 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon_nsr \
    --momentum 0.98 \
    --muon_eps 1e-15 \
    --muon_nsr_alpha 1000.0 \
    --save_dir checkpoints/llama_300m_4096seq \
    --name llama_300m_muon_nsr \
    --wandb_project "stf-igcai-300M" \
    --wandb_name "muon_nsr-alpha=1000" \
    --dataset_info data/dataset_info_llama2.json

# Muon_VS optimizer for Llama 300M
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_300m.json \
    --lr 8e-3 \
    --adamw_lr_multiplier 0.3 \
    --batch_size 16 \
    --total_batch_size 128 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 11500 \
    --warmup_steps 0 \
    --stable_steps 2300 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 100 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon_vs \
    --momentum 0.98 \
    --muon_eps 1e-15 \
    --muon_vs_alpha 1000.0 \
    --save_dir checkpoints/llama_300m_4096seq \
    --name llama_300m_muon_vs \
    --wandb_project "stf-igcai-300M" \
    --wandb_name "muon_vs" \
    --dataset_info data/dataset_info_llama2.json


# ============================================================================
# Llama 1.2B Experiments
# ============================================================================

# Muon optimizer for Llama 1.2B
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_1.2b.json \
    --lr 8e-3 \
    --adamw_lr_multiplier 0.15 \
    --batch_size 8 \
    --total_batch_size 256 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 23000 \
    --warmup_steps 0 \
    --stable_steps 0 \
    --weight_decay 0.1 \
    --grad_clipping 2.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 250 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon \
    --momentum 0.98 \
    --muon_eps 1e-15 \
    --save_dir checkpoints/llama_1.2b_4096seq \
    --name llama_1.2b_muon \
    --wandb_project "stf-igcai-1.2b" \
    --wandb_name "muon" \
    --dataset_info data/dataset_info_llama2.json

# Muon_NSR optimizer for Llama 1.2B
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_1.2b.json \
    --lr 8e-3 \
    --adamw_lr_multiplier 0.15 \
    --batch_size 8 \
    --total_batch_size 256 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 23000 \
    --warmup_steps 0 \
    --stable_steps 0 \
    --weight_decay 0.1 \
    --grad_clipping 2.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 250 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon_nsr \
    --momentum 0.98 \
    --muon_eps 1e-15 \
    --muon_nsr_alpha 1000.0 \
    --save_dir checkpoints/llama_1.2b_4096seq \
    --name llama_1.2b_muon_nsr \
    --wandb_project "stf-igcai-1.2b" \
    --wandb_name "muon_nsr_alpha_1000.0" \
    --dataset_info data/dataset_info_llama2.json
 
# Muon_VS optimizer for Llama 1.2B
torchrun --standalone --nproc_per_node 8 train_Llama.py \
    --model_config configs/llama_1.2b.json \
    --lr 8e-3 \
    --adamw_lr_multiplier 0.15 \
    --batch_size 8 \
    --total_batch_size 256 \
    --max_length 4096 \
    --activation_checkpointing \
    --num_training_steps 23000 \
    --warmup_steps 0 \
    --stable_steps 0 \
    --weight_decay 0.1 \
    --grad_clipping 2.0 \
    --betas 0.8 0.98 \
    --dtype bfloat16 \
    --eval_every 250 \
    --scheduler linear \
    --min_lr_ratio 0 \
    --optimizer muon_vs \
    --momentum 0.98 \
    --muon_eps 1e-15 \
    --muon_vs_alpha 1000.0 \
    --save_dir checkpoints/llama_1.2b_4096seq \
    --name llama_1.2b_muon_vs \
    --wandb_project "stf-igcai-1.2b" \
    --wandb_name "muon_vs" \
    --dataset_info data/dataset_info_llama2.json