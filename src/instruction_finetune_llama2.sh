#!/bin/bash


CUDA_VISIBLE_DEVICES="0,1,2,3"


acclerate launch python train.py 
	--data-path 
	--model-path
	--config-path
	--logging-path


