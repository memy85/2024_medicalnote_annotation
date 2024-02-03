#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:30
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7

echo "cuda devices : $CUDA_VISIBLE_DEVICES"

python3 test_mistral_fewshot.py
