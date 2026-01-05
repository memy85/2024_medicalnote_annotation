#!/bin/bash

# We will use accelerate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true
# MODEL_NAME=biomistral7b_avigon
#

####################### Fine-tuning scripts

# for MODEL_NAME in  'mistral7b'  ; do
# # for MODEL_NAME in 'llama31' ; do
# 	for CV_IDX in 0 1 2 3 4 5 6 7 8 9  ; do 
# 		accelerate launch --multi_gpu --num_processes 2 \
# 		src/finetune_models.py \
# 		--model_name_or_path $MODEL_NAME \
# 		--data_path data/processed/finetune/cv${CV_IDX}_generic.json \
# 		--bf16 true \
# 		--use_fast_tokenizer False \
# 		--output_dir ./models/${MODEL_NAME}-finetuned-${CV_IDX}  \
# 		--logging_dir ./logs/models/${MODEL_NAME} \
# 		--num_train_epochs 5 \
# 		--gradient_accumulation_steps 1 \
# 		--per_device_train_batch_size 2  \
# 		--model_max_length 8192 \
# 		--save_strategy steps  \
# 		--save_steps 10  \
# 		--logging_steps 10 \
# 		--save_total_limit 3 \
# 		--learning_rate 3e-4  \
# 		--weight_decay 0.01  \
# 		--warmup_steps 400  \
# 		--do_lora True  \
# 		--lora_r 64  \
# 		--report_to tensorboard  \
# 		--lora_target_modules "q_proj,v_proj,k_proj,o_proj"  \
# 		--gradient_checkpointing True \
# 		--load_in_8bit False \
# 		--deepspeed config/ds_config.json
# 	done 
# done

# for MODEL_NAME in 'llama31' ; do
# # for MODEL_NAME in 'llama31' ; do
# 	for CV_IDX in 0 1 2 3 4 5 6 7 8 9  ; do 
# 		accelerate launch --multi_gpu --num_processes 2 \
# 		src/finetune_models.py \
# 		--model_name_or_path $MODEL_NAME \
# 		--data_path data/processed/finetune/cv${CV_IDX}_structured.json \
# 		--bf16 true \
# 		--use_fast_tokenizer False \
# 		--output_dir ./models/${MODEL_NAME}-finetuned-${CV_IDX}  \
# 		--logging_dir ./logs/models/${MODEL_NAME} \
# 		--num_train_epochs 5 \
# 		--gradient_accumulation_steps 1 \
# 		--per_device_train_batch_size 2  \
# 		--model_max_length 8192 \
# 		--save_strategy steps  \
# 		--save_steps 10  \
# 		--logging_steps 10 \
# 		--save_total_limit 3 \
# 		--learning_rate 3e-4  \
# 		--weight_decay 0.01  \
# 		--warmup_steps 400  \
# 		--do_lora True  \
# 		--lora_r 64  \
# 		--report_to tensorboard  \
# 		--lora_target_modules "q_proj,v_proj,k_proj,o_proj"  \
# 		--gradient_checkpointing True \
# 		--load_in_8bit False \
# 		--deepspeed config/ds_config.json
# 	done 
# done

####################### Measure Scores

# for MODEL_NAME in 'llama31' 'biomistral7b_avigon' ; do
# 	for CV_IDX in 0 1 2 3 4 5 6 7 8 9 ; do
# 		python3 src/test_finetuned_model.py --model $MODEL_NAME --finetune True --shot zeroshot --cv_idx $CV_IDX
# 	done
# done
for MODEL_NAME in 'mistral7b' 'deepseek'; do
	for CV_IDX in 0 1 2 3 4 5 6 7 8 9 ; do
		python3 src/test_finetuned_model.py --model $MODEL_NAME --finetune True --shot zeroshot --cv_idx $CV_IDX
	done
done
