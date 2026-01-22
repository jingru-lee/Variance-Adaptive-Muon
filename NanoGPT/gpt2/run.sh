# The project path
export ROOT="./gpt2"

# NanaoGPT Small

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/small_adamw.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/small_muon.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/small_adamuon.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/small_muon_nsr.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/small_muon_vs.py


# NanaoGPT Medium

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/medium_adamw.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/medium_muon.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/medium_adamuon.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/medium_muon_nsr.py

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/medium_muon_vs.py


      