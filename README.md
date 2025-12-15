# Commands


## Local run.sh
python main.py \
  --decoder gpt2 \
  --train_bs 32 \
  --eval_bs 32 \
  --num_samples_recall_train 50 \
  --num_samples_rerank_train 50 \
  --eval_every 10000 \
  --check_learned_weights False \
  --load_model_path Outputs/REDIAL/temp/CRS_Train_checkpoint_epoch_9 \
  --save True"



# PECRS
Source code for the paper *Parameter-Efficient Conversational Recommender System as a Language Processing Task*.

Mathieu Ravaut, Hao Zhang, Lu Xu, Aixin Sun, Yong Liu.

Accepted for publication at EACL 2024. 

This repo is built upon [efficient_unified_crs](https://github.com/Ravoxsg/efficient_unified_crs)

