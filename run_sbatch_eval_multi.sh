#!/bin/bash


# ==========================
#  EVAL INSPIRED EPOCH 10
# ==========================
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=PECRS_eval_e10_INSPIRED
#SBATCH --output=logs/PECRS_eval_e10_INSPIRED_%j.out
#SBATCH --error=logs/PECRS_eval_e10_INSPIRED_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

singularity exec --nv \
  -B ./my_venv:/scratch/my_venv \
  /ceph/container/python/python_3.8.sif \
  /bin/bash -c "
    source /scratch/my_venv/bin/activate && \
    python main.py \
      --mode eval \
      --decoder gpt2 \
      --dataset_name INSPIRED \
      --exp_name eval_epoch_10 \
      --eval_bs 32 \
      --check_learned_weights True \
      --load_model_path Outputs/INSPIRED/temp/CRS_Train_checkpoint_epoch_10
  "
EOF

# ==========================
#  EVAL REDIAL EPOCH 10
# ==========================
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=PECRS_eval_e10_REDIAL
#SBATCH --output=logs/PECRS_eval_e10_REDIAL_%j.out
#SBATCH --error=logs/PECRS_eval_e10_REDIAL_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

singularity exec --nv \
  -B ./my_venv:/scratch/my_venv \
  /ceph/container/python/python_3.8.sif \
  /bin/bash -c "
    source /scratch/my_venv/bin/activate && \
    python main.py \
      --mode eval \
      --decoder gpt2 \
      --dataset_name REDIAL \
      --exp_name eval_epoch_10 \
      --eval_bs 32 \
      --check_learned_weights True \
      --load_model_path Outputs/REDIAL/temp/CRS_Train_checkpoint_epoch_10
  "
EOF
