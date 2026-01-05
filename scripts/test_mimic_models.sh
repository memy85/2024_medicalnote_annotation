#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=true

# MODEL='llama31'

# ------- YOU NEED NOT CHANGE BELOW
# ------- ONLY CHANGE THE ABOVE

FINETUNE='true'
SHOT='zeroshot'
# PROMPT_NAME='generic'

# INFERENCE
# for MODEL in 'llama31' 'mistral7b' 'biomistral7b_avigon' 'deepseek' ; do
# 	for AUG in 10 100 1000 10000; do
# 		# for CV_IDX in $(seq 0 9);
# 		# do
# 		# 	echo $CV_IDX
# 		python3 src/test_mimic_model.py --model $MODEL  --aug_size $AUG
# 		# done
#
# 		# AGGREGATE RESULTS
# 		echo "$MODEL and $AUG " 
# 		python3 src/aggregate_mimic_scores.py --model $MODEL  --aug_size $AUG 
# 	done
# done


for MODEL in 'deepseek' ; do
	for AUG in 10000; do
		python3 src/test_mimic_model.py --model $MODEL  --aug_size $AUG

		# AGGREGATE RESULTS
		echo "$MODEL and $AUG " 
		python3 src/aggregate_mimic_scores.py --model $MODEL  --aug_size $AUG 
	done
done

