#!/bin/sh
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

#export CUDA_HOME="/usr/local/cuda-11.3"
#export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
#export PATH="/usr/local/cuda-11.3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2 #""

python3 src/evaluation.py \
    --model_name_or_path /data/data_user_alpha/public_models/Mixtral-8x7B-Instruct-v0.1 \
    --train_file /home/zhichaoyang/pubmedgpt_ct_data/data_processed/usmleqa/train.json \
    --validation_file /home/zhichaoyang/pubmedgpt_ct_data/data_processed/usmleqa/valid.json \
    --test_file /home/zhichaoyang/pubmedgpt_ct_data/data_processed/usmleqa/test.json \
    --max_new_tokens 2 \
    --template alpaca \
    --finetuning_type lora \
    --quantization_bit 4 \
#    --checkpoint_dir output/mistral_instruct_4bit/checkpoint-15600