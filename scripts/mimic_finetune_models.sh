#!/bin/bash

# We will use accelerate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export N_GPU=1
export TOKENIZERS_PARALLELISM=true
# MODEL_NAME=biomistral7b_avigon

# for GPU in 0 ; do 
# GPU=0,1,2,3,4
# --gpu_ids 0,1,2,3,4 \
# 'llama31' first 10 100
# AUG_SIZES=(10 100 1000 10000)
AUG_SIZES=(10000)
NUM_EPOCHS=(2)

# for MODEL_NAME in 'biomistral7b_avigon' 'mistral7b' 'deepseek' ; do
for MODEL_NAME in 'deepseek' ; do
	for i in "${!AUG_SIZES[@]}" ; do 
	# for AUG_SIZE in 10000 ; do 
		AUG_SIZE=${AUG_SIZES[i]}
		EPOCHS=${NUM_EPOCHS[i]}
		# accelerate launch --multi_gpu --num_processes $N_GPU \
		accelerate launch --num_processes $N_GPU \
		--config_file config/single_machine_config.yaml \
		src/finetune_models.py \
		--model_name_or_path ${MODEL_NAME} \
		--data_path data/processed/mad${AUG_SIZE}_generic.json \
		--bf16 true \
		--use_fast_tokenizer False \
		--output_dir ./models/${MODEL_NAME}-mimic-finetuned-${AUG_SIZE}  \
		--logging_dir ./logs/models/${MODEL_NAME}-mimic-finetuned-${AUG_SIZE} \
		--num_train_epochs $EPOCHS \
		--gradient_accumulation_steps 1 \
		--per_device_train_batch_size 1  \
		--model_max_length 8192 \
		--save_strategy steps  \
		--save_steps 10  \
		--logging_steps 10 \
		--save_total_limit 3 \
		--learning_rate 3e-4  \
		--weight_decay 0.001  \
		--warmup_steps 200  \
		--do_lora True  \
		--lora_r 64  \
		--report_to tensorboard  \
		--lora_target_modules "q_proj,v_proj,k_proj,o_proj"  \
		--gradient_checkpointing True \
		--load_in_8bit False \
		# --deepspeed config/single_machine_config.yaml
		# --deepspeed config/ds_config.json
		# --lora_weight_path ./models/${MODEL_NAME}-mimic-finetuned-${AUG_SIZE} \
	done 
		python3 -c "from src.utils import *; send_line_message('finished for ${MODEL_NAME}')"
done

AUG_SIZES=(1000 10000)
NUM_EPOCHS=(4 2)

for MODEL_NAME in 'llama31'  ; do
	for i in "${!AUG_SIZES[@]}" ; do 
		AUG_SIZE=${AUG_SIZES[i]}
		EPOCHS=${NUM_EPOCHS[i]}
		# accelerate launch --multi_gpu --num_processes $N_GPU \
		accelerate launch --num_processes $N_GPU \
		--config_file config/single_machine_config.yaml \
		src/finetune_models.py \
		--model_name_or_path ${MODEL_NAME} \
		--data_path data/processed/mad${AUG_SIZE}_structured.json \
		--bf16 true \
		--use_fast_tokenizer False \
		--output_dir ./models/${MODEL_NAME}-mimic-finetuned-${AUG_SIZE}  \
		--logging_dir ./logs/models/${MODEL_NAME}-mimic-finetuned-${AUG_SIZE} \
		--num_train_epochs $EPOCHS \
		--gradient_accumulation_steps 1 \
		--per_device_train_batch_size 1  \
		--model_max_length 8192 \
		--save_strategy steps  \
		--save_steps 10  \
		--logging_steps 10 \
		--save_total_limit 3 \
		--learning_rate 3e-4  \
		--weight_decay 0.01  \
		--warmup_steps 200  \
		--do_lora True  \
		--lora_r 64  \
		--report_to tensorboard  \
		--lora_target_modules "q_proj,v_proj,k_proj,o_proj"  \
		--gradient_checkpointing True \
		--load_in_8bit False \
		--resume_from_checkpoint True \
		# --deepspeed config/single_machine_config.yaml
		# --deepspeed config/ds_config.json
		# --lora_weight_path ./models/${MODEL_NAME}-mimic-finetuned-${AUG_SIZE} \
	done 
	python3 -c "from src.utils import *; send_line_message('finished for ${MODEL_NAME}') "
done
