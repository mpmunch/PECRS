#!/bin/bash

#SBATCH --job-name=PECRS_training
#SBATCH --output=PECRS_training.out
#SBATCH --error=PECRS_training.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Run script in container
singularity exec --nv \
 -B ./my_venv:/scratch/my_venv \
 /ceph/container/python/python_3.8.sif \
  /bin/bash -c "source /scratch/my_venv/bin/activate && python main.py \
  --decoder gpt2 \
  --train_bs 32 \
  --eval_bs 64 \
  --num_samples_recall_train 50 \
  --num_samples_rerank_train 50 \
  --eval_every 10000 \
  --check_learned_weights False \
  --load_model_path Outputs/REDIAL/temp/CRS_Train_Last.pt \
  --save True"
