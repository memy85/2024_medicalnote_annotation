#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:30
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

echo "cuda devices : $CUDA_VISIBLE_DEVICES"

# zeroshot 7b
python3 src/test_mistral.py \
                --model 7b \
                --inference zeroshot \
                --topn top3 \
                --save_name top3_zeroshot_results \
                --max_new_token 50 \
                &> zeroshot_error.log

# fewshot 7b
python3 src/test_mistral.py \
               --model 7b \
               --inference fewshot \
               --topn top3 \
               --save_name top3_fewshot_results \
               --max_new_token 50 \
               &> fewshot_error.log

# zeroshot 7b top5
python3 src/test_mistral.py \
                --model 7b \
                --inference zeroshot \
                --topn top5 \
                --save_name top5_zeroshot_results \
                --max_new_token 100 \
                &> logs/zeroshot_error.log

# fewshot 7b top5
python3 src/test_mistral.py \
                --model 7b \
                --inference fewshot \
                --topn top5 \
                --save_name top5_fewshot_results \
                --max_new_token 100 \
                &> logs/fewshot_top5_error.log

# zeroshot 7b top10
python3 src/test_mistral.py \
                --model 7b \
                --inference zeroshot \
                --topn top10 \
                --save_name top10_zeroshot_results \
                --max_new_token 250 \
                &> logs/zeroshot_error.log

# fewshot 7b top10
python3 src/test_mistral.py \
                --model 7b \
                --inference fewshot \
                --topn top10 \
                --save_name top10_fewshot_results \
                --max_new_token 250 \
                &> logs/fewshot_error.log

# ============================= =============== ===============================
# ============================= Automodel Codes =============================== 
# ============================= =============== ===============================

# fewshot-automodel 7b
# python3 test_mistral_v2.py \
#                 --model 7b \
#                 --inference fewshot \
#                 --instruction top3 \
#                 --save_name top3_fewshot_results_7b \
#                 &> fewshot_error.log
