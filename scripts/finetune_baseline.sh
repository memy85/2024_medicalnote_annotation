#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# export TOKENIZERS_PARALLELISM=true


# ------- YOU NEED NOT CHANGE BELOW
# ------- ONLY CHANGE THE ABOVE

# RUN CODE
for MODEL in 'biogpt'; do 
	python3 src/train_baseline.py --model $MODEL 
done

