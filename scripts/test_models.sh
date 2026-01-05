#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# export TOKENIZERS_PARALLELISM=true

# MODEL='mistral7b'

# ------- YOU NEED NOT CHANGE BELOW
# ------- ONLY CHANGE THE ABOVE

FINETUNE='true'
SHOT='zeroshot'
PROMPT_NAME='generic'

# INFERENCE
# for CV_IDX in $(seq 0 9);
# do
# 	echo $CV_IDX
# 	python3 src/test_model.py --model $MODEL --shot $SHOT --finetune $FINETUNE  --prompt_name $PROMPT_NAME --cv_idx $CV_IDX
# done

# AGGREGATE RESULTS
for MODEL in 'llama31' 'mistral7b' 'biomistral7b_avigon' 'deepseek'; do
	python3 src/aggregate_scores.py --model $MODEL --shot $SHOT --finetune $FINETUNE --prompt_name $PROMPT_NAME 
done


