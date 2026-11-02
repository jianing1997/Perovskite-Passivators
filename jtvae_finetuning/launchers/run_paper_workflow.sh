#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"


# Paper setting: ten rounds, 100 selected candidates per round, 1 initial epoch,
# then 5 epochs per subsequent round; 1,000 monitoring samples per round.
export CUDA_VISIBLE_DEVICES=0

python -m jtvae_finetuning.workflow.opt_chem \
  --seed=1 --gpu \
  --query_budget=1000 \
  --retraining_frequency=100 \
  --samples_per_model=1000 \
  --result_root=logs/DN_DM_epochs5/rank_pf/k_1e-05/r_100/seed1 \
  --pretrained_model_file=runtime_assets/pretrained_models/chem.ckpt \
  --train_path=runtime_assets/data/chem/zinc/orig_model/tensors_train \
  --val_path=runtime_assets/data/chem/zinc/orig_model/tensors_val \
  --vocab_file=runtime_assets/data/chem/zinc/orig_model/vocab.txt \
  --property_file=runtime_assets/data/chem/zinc/orig_model/dn_all.pkl \
  --property2_file=runtime_assets/data/chem/zinc/orig_model/dm_all.pkl \
  --property=DN \
  --property2=DM \
  --n_retrain_epochs=5 \
  --n_init_retrain_epochs=1 \
  --weight_type=rank_pf \
  --rank_weight_k=1e-05 \
  --all_new=1

