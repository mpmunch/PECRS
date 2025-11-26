srun --mem=24G --cpus-per-task=15 --gres=gpu:1 --time=12:00:00 singularity exec --nv \
 -B ./my_venv:/scratch/my_venv \
 /ceph/container/python/python_3.8.sif \
  /bin/bash -c "source /scratch/my_venv/bin/activate && python main.py \
  --decoder gpt2 \
  --train_bs 32 \
  --num_samples_recall_train 50 \
  --num_samples_rerank_train 50 \
  --eval_every 10000 \
  --check_learned_weights False \
  --load_model_path Outputs/REDIAL/temp/CRS_Train_Last.pt \
  --save True"

