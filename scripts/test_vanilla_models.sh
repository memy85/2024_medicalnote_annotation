#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=true

# MODEL='deepseek'

# ------- YOU NEED NOT CHANGE BELOW
# ------- ONLY CHANGE THE ABOVE

FINETUNE='false'

# INFERENCE
for MODEL in 'llama31' 'mistral7b' 'biomistral7b_avigon' 'deepseek'; do
	python3 src/test_vanilla_model.py --model $MODEL 
done

# AGGREGATE RESULTS
# for MODEL in 'llama31' 'mistral7b' 'biomistral7b_avigon' 'deepseek'; do
# 	for SHOT in 'zeroshot' 'fewshot'; do 
# 		for PROMPT in 'generic' 'structured'; do
# 		python3 src/aggregate_scores.py --model $MODEL --shot $SHOT --finetune 'false' --prompt_name $PROMPT
# 		done
# 	done
# done


