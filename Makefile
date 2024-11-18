
CUDA_DEVICE_ORDER := PCI_BUS_ID
CUDA_LAUNCH_BLOCKING := 1
CUDA_VISIBLE_DEVICES := 0,1
TORCH_EXTENSIONS_DIR :=/home/zhichaoyang/.cache/torch_extensions

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export TORCH_EXTENSIONS_DIR



finetune_mistral_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path mistral7b \
	  --data_path data/processed/discharge_dataset.json \
	  --bf16 False \
	  --use_fast_tokenizer False \
	  --output_dir ./models/mistral7b_mimic \
	  --log_level debug \
	  --logging_dir ./logs/models/mistral7b_mimic/logs \
	  --num_train_epochs 100 \
	  --per_device_train_batch_size 1 \
	  --gradient_accumulation_steps 128 \
	  --model_max_length 20000 \
	  --evaluation_strategy "no" \
	  --save_strategy "steps" \
	  --save_steps 10 \
	  --logging_steps 1 \
	  --save_total_limit 100 \
	  --learning_rate 3e-4 \
	  --weight_decay 0.0 \
	  --warmup_steps 200 \
	  --do_lora True \
	  --lora_r 64 \
	  --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
	  --gradient_checkpointing True \
	  --load_in_8bit False \
	  --report_to tensorboard \
	  > logs/finetune_mistral7b_on_mimic.log


finetune_biomistral_avigon_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path biomistral7b_avigon \
	  --data_path data/processed/discharge_dataset.json \
	  --bf16 False \
	  --use_fast_tokenizer False \
	  --output_dir ./models/biomistral7b_avigon_mimic \
	  --log_level debug \
	  --logging_dir ./logs/models/biomistral7b_avigon_mimic/logs \
	  --num_train_epochs 100 \
	  --per_device_train_batch_size 1 \
	  --gradient_accumulation_steps 128 \
	  --model_max_length 20000 \
	  --evaluation_strategy "no" \
	  --save_strategy "steps" \
	  --save_steps 10 \
	  --logging_steps 1 \
	  --save_total_limit 100 \
	  --learning_rate 3e-4 \
	  --weight_decay 0.0 \
	  --warmup_steps 200 \
	  --do_lora True \
	  --lora_r 64 \
	  --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
	  --gradient_checkpointing True \
	  --load_in_8bit False \
	  --report_to tensorboard \
	  > logs/finetune_biomistral7b_avigon_on_mimic.log
	  
finetune_mistral_modified_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path mistral7b \
	  --data_path data/processed/modified_discharge_dataset.json \
	  --bf16 False \
	  --use_fast_tokenizer False \
	  --output_dir ./models/mistral7b_modified_mimic \
	  --log_level debug \
	  --logging_dir ./logs/models/mistral7b_mimic/logs \
	  --num_train_epochs 100 \
	  --per_device_train_batch_size 1 \
	  --gradient_accumulation_steps 128 \
	  --model_max_length 20000 \
	  --evaluation_strategy "no" \
	  --save_strategy "steps" \
	  --save_steps 10 \
	  --logging_steps 1 \
	  --save_total_limit 100 \
	  --learning_rate 3e-4 \
	  --weight_decay 0.0 \
	  --warmup_steps 200 \
	  --do_lora True \
	  --lora_r 64 \
	  --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
	  --gradient_checkpointing True \
	  --load_in_8bit False \
	  --report_to tensorboard \
	  > logs/finetune_mistral7b_on_mimic.log


finetune_biomistral_avigon_modified_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path biomistral7b_avigon \
	  --data_path data/processed/modified_discharge_dataset.json \
	  --bf16 False \
	  --use_fast_tokenizer False \
	  --output_dir ./models/biomistral7b_avigon_modified_mimic \
	  --log_level debug \
	  --logging_dir ./logs/models/biomistral7b_avigon_mimic/logs \
	  --num_train_epochs 100 \
	  --per_device_train_batch_size 1 \
	  --gradient_accumulation_steps 128 \
	  --model_max_length 20000 \
	  --evaluation_strategy "no" \
	  --save_strategy "steps" \
	  --save_steps 10 \
	  --logging_steps 1 \
	  --save_total_limit 100 \
	  --learning_rate 3e-4 \
	  --weight_decay 0.0 \
	  --warmup_steps 200 \
	  --do_lora True \
	  --lora_r 64 \
	  --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
	  --gradient_checkpointing True \
	  --load_in_8bit False \
	  --report_to tensorboard \
	  > logs/finetune_biomistral7b_avigon_on_mimic.log


finetune_llama31_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path llama31 \
	  --data_path data/processed/discharge_dataset.json \
	  --bf16 False \
	  --use_fast_tokenizer False \
	  --output_dir ./models/llama31_mimic \
	  --log_level debug \
	  --logging_dir ./logs/models/llama31_mimic/logs \
	  --num_train_epochs 100 \
	  --per_device_train_batch_size 1 \
	  --gradient_accumulation_steps 128 \
	  --model_max_length 20000 \
	  --evaluation_strategy "no" \
	  --save_strategy "steps" \
	  --save_steps 10 \
	  --logging_steps 1 \
	  --save_total_limit 100 \
	  --learning_rate 3e-4 \
	  --weight_decay 0.0 \
	  --warmup_steps 200 \
	  --do_lora True \
	  --lora_r 64 \
	  --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
	  --gradient_checkpointing True \
	  --load_in_8bit False \
	  --report_to tensorboard \
	  > logs/finetune_llama31_on_mimic.log 2>&1


finetune_biomistral7b_increment : 
	for number in 1000 ; do \
		python3 src/finetune_models.py \
		--model_name_or_path biomistral7b_avigon \
		--data_path data/processed/modified_discharge_dataset_augmented_$$number.json \
		--bf16 False \
		--use_fast_tokenizer False \
		--output_dir ./models/biomistral7b_avigon_augmented_modified_$$number \
		--log_level debug \
		--logging_dir ./logs/models/biomistral7b_avigon_mimic/logs \
		--num_train_epochs 60 \
		--per_device_train_batch_size 1 \
		--gradient_accumulation_steps 128 \
		--model_max_length 20000 \
		--evaluation_strategy "no" \
		--save_strategy "steps" \
		--save_steps 10 \
		--logging_steps 1 \
		--save_total_limit 100 \
		--learning_rate 3e-4 \
		--weight_decay 0.0 \
		--warmup_steps 200 \
		--do_lora True \
		--lora_r 64 \
		--lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
		--gradient_checkpointing True \
		--load_in_8bit False \
		--report_to tensorboard \
		> logs/finetune_biomistral7b_avigon_on_mimic.log ;\
	done

finetune_biomistral7b_avigon_modified : 
	python3 src/finetune_models.py \
	--model_name_or_path biomistral7b_avigon \
	--data_path data/processed/modified_discharge_dataset.json \
	--bf16 False \
	--use_fast_tokenizer False \
	--output_dir ./models/biomistral7b_avigon_augmented_modified_$$number \
	--log_level debug \
	--logging_dir ./logs/models/biomistral7b_avigon_modified/logs \
	--num_train_epochs 60 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 128 \
	--model_max_length 20000 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 10 \
	--logging_steps 1 \
	--save_total_limit 100 \
	--learning_rate 3e-4 \
	--weight_decay 0.0 \
	--warmup_steps 200 \
	--do_lora True \
	--lora_r 64 \
	--lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
	--gradient_checkpointing True \
	--load_in_8bit False \
	--report_to tensorboard \
	> logs/finetune_biomistral7b_avigon_modified.log ;\


#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- Cross-Validation

cv_no_finetune :
	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top3 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top3 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top3_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top3 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top3 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top5 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top5 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top5_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top5 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top5 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top10 --save_name mistral7b --max_new_token 400 > ./logs/mistral7b_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top10 --save_name mistral7b --max_new_token 400 > ./logs/mistral7b_top10_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top10 --save_name biomistral7b_avigon --max_new_token 400 > ./logs/biomistral7b_avigon_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top10 --save_name biomistral7b_avigon --max_new_token 400 > ./logs/biomistral7b_avigon_top10_fewshot.log 2>&1

cv_finetune_rank3 :
	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top3 --save_name mistral7b_finetune --max_new_token 200 > ./logs/mistral7b_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top3 --save_name mistral7b_finetune --max_new_token 200 > ./logs/mistral7b_finetune_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b --finetune True --inference zeroshot --topn top3 --save_name biomistral7b_finetune --max_new_token 200 > ./logs/biomistral7b_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b --finetune True --inference fewshot --topn top3 --save_name biomistral7b_finetune --max_new_token 200 > ./logs/biomistral7b_finetune_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top3 --save_name biomistral7b_avigon_finetune --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top3 --save_name biomistral7b_avigon_finetune --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top3_fewshot.log 2>&1

cv_finetune_rank5 :
	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top5 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_finetune_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top5 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_finetune_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b --finetune True --inference zeroshot --topn top5 --save_name biomistral7b --max_new_token 200 > ./logs/biomistral7b_finetune_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b --finetune True --inference fewshot --topn top5 --save_name biomistral7b --max_new_token 200 > ./logs/biomistral7b_finetune_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top5 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top5 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top5_fewshot.log 2>&1

cv_finetune_rank10 :
	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top10 --save_name mistral7b --max_new_token 400 > ./logs/mistral7b_finetune_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top10 --save_name mistral7b --max_new_token 400 > ./logs/mistral7b_finetune_top10_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b --finetune True --inference zeroshot --topn top10 --save_name biomistral7b --max_new_token 400 > ./logs/biomistral7b_finetune_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b --finetune True --inference fewshot --topn top10 --save_name biomistral7b --max_new_token 400 > ./logs/biomistral7b_finetune_top10_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top10 --save_name biomistral7b_avigon --max_new_token 400 > ./logs/biomistral7b_avigon_finetune_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top10 --save_name biomistral7b_avigon --max_new_token 400 > ./logs/biomistral7b_avigon_finetune_top10_fewshot.log 2>&1


cv_finetune_all : cv_finetune_rank3 cv_finetune_rank5 cv_finetune_rank10


cv_finetuned_rank3 :
	python3 src/cross_validation.py --model mistral7b --finetune True --just_evaluation True --inference zeroshot --topn top3 --save_name mistral7b_finetune --max_new_token 200 > ./logs/mistral7b_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --just_evaluation True --inference fewshot --topn top3 --save_name mistral7b_finetune --max_new_token 200 > ./logs/mistral7b_finetune_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --just_evaluation True --inference zeroshot --topn top3 --save_name biomistral7b_avigon_finetune --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --just_evaluation True --inference fewshot --topn top3 --save_name biomistral7b_avigon_finetune --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top3_fewshot.log 2>&1

cv_finetuned_rank5 :
	python3 src/cross_validation.py --model mistral7b --finetune True --just_evaluation True --inference zeroshot --topn top5 --save_name mistral7b_finetune --max_new_token 200 > ./logs/mistral7b_finetune_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --just_evaluation True --inference fewshot --topn top5 --save_name mistral7b_finetune --max_new_token 200 > ./logs/mistral7b_finetune_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --just_evaluation True --inference zeroshot --topn top5 --save_name biomistral7b_avigon_finetune --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --just_evaluation True --inference fewshot --topn top5 --save_name biomistral7b_avigon_finetune --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top5_fewshot.log 2>&1

cv_finetuned_rank10 :
	python3 src/cross_validation.py --model mistral7b --finetune True --just_evaluation True --inference zeroshot --topn top10 --save_name mistral7b_finetune --max_new_token 400 > ./logs/mistral7b_finetune_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --just_evaluation True --inference fewshot --topn top10 --save_name mistral7b_finetune --max_new_token 400 > ./logs/mistral7b_finetune_top10_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --just_evaluation True --inference zeroshot --topn top10 --save_name biomistral7b_avigon_finetune --max_new_token 400 > ./logs/biomistral7b_avigon_finetune_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --just_evaluation True --inference fewshot --topn top10 --save_name biomistral7b_avigon_finetune --max_new_token 400 > ./logs/biomistral7b_avigon_finetune_top10_fewshot.log 2>&1


cv_finetuned_all : cv_finetuned_rank3 cv_finetuned_rank5 cv_finetuned_rank10


cv_mistral_mimic_finetune :
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top3 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top3 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top5 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top5 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top10 --save_name mistral7b_mimic --max_new_token 400 > ./logs/mistral7b_mimic_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top10 --save_name mistral7b_mimic --max_new_token 400 > ./logs/mistral7b_mimic_top10_fewshot.log 2>&1

cv_biomistral_mimic_finetune :
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top3 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top3 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top5 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top5 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top10 --save_name biomistral7b_mimic --max_new_token 400 > ./logs/biomistral7b_mimic_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top10 --save_name biomistral7b_mimic --max_new_token 400 > ./logs/biomistral7b_mimic_top10_fewshot.log 2>&1


modified_cv_finetune :
	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top3 --save_name mistral7b_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top3 --save_name mistral7b_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_top3_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top3 --save_name biomistral7b_avigon_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top3 --save_name biomistral7b_avigon_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top5 --save_name mistral7b_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top5 --save_name mistral7b_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_top5_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top5 --save_name biomistral7b_avigon_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top5 --save_name biomistral7b_avigon_modified_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top10 --save_name mistral7b_modified_finetune --max_new_token 500 --prompt_name modified > ./logs/mistral7b_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top10 --save_name mistral7b_modified_finetune --max_new_token 500 --prompt_name modified > ./logs/mistral7b_top10_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top10 --save_name biomistral7b_avigon_modified_finetune --max_new_token 500 --prompt_name modified > ./logs/biomistral7b_avigon_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top10 --save_name biomistral7b_avigon_modified_finetune --max_new_token 500 --prompt_name modified > ./logs/biomistral7b_avigon_top10_fewshot.log 2>&1

	python3 -c "from src.utils import * ; send_line_message('finished modified_cv_finetune top 10');"

modified_cv_mimic_finetune :
	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top3 --save_name mistral7b_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top3 --save_name mistral7b_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_modified --finetune False --inference zeroshot --topn top3 --save_name biomistral7b_avigon_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_modified --finetune False --inference fewshot --topn top3 --save_name biomistral7b_avigon_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_modified_mimic_finetune.log 2>&1

	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top5 --save_name mistral7b_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top5 --save_name mistral7b_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/mistral7b_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_modified --finetune False --inference zeroshot --topn top5 --save_name biomistral7b_avigon_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_modified --finetune False --inference fewshot --topn top5 --save_name biomistral7b_avigon_modified_mimic_finetune --max_new_token 200 --prompt_name modified > ./logs/biomistral7b_avigon_modified_mimic_finetune.log 2>&1

	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top10 --save_name mistral7b_modified_mimic_finetune --max_new_token 500 --prompt_name modified > ./logs/mistral7b_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top10 --save_name mistral7b_modified_mimic_finetune --max_new_token 500 --prompt_name modified > ./logs/mistral7b_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_modified --finetune False --inference zeroshot --topn top10 --save_name biomistral7b_avigon_modified_mimic_finetune --max_new_token 500 --prompt_name modified > ./logs/biomistral7b_avigon_modified_mimic_finetune.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon_mimic_modified --finetune False --inference fewshot --topn top10 --save_name biomistral7b_avigon_modified_mimic_finetune --max_new_token 500 --prompt_name modified > ./logs/biomistral7b_avigon_modified_mimic_finetune.log 2>&1

baseline_cv :
	# python3 src/test_baseline.py --model gpt --finetune True --topn top3 --save_name gpt --max_new_token 200 > ./logs/gpt_top3.log 2>&1
	# python3 src/test_baseline.py --model gpt --finetune True --topn top5 --save_name gpt --max_new_token 200 > ./logs/gpt_top5.log 2>&1
	# python3 src/test_baseline.py --model gpt --finetune True --topn top10 --save_name gpt --max_new_token 400 > ./logs/gpt_top10.log 2>&1

	python3 src/test_baseline.py --model biogpt --finetune True --topn top3 --save_name biogpt --max_new_token 200 > ./logs/biogpt_top3.log 2>&1
	python3 src/test_baseline.py --model biogpt --finetune True --topn top5 --save_name biogpt --max_new_token 200 > ./logs/biogpt_top5.log 2>&1
	python3 src/test_baseline.py --model biogpt --finetune True --topn top10 --save_name biogpt --max_new_token 400 > ./logs/biogpt_top10.log 2>&1

cv_biomistral_modified_mimic_all_cases :
	for number in 1000 ; do \
		for ranks in 3 5 10 ; do \
			python3 src/cross_validation.py --model biomistral7b_avigon_modified_mimic$$number --finetune False --inference zeroshot --topn top$$ranks --save_name biomistral7b_avigon_modified_mimic$$number --max_new_token 300 --prompt_name modified > ./logs/biomistral7b_avigon_modified.log 2>&1; \
			python3 src/cross_validation.py --model biomistral7b_avigon_modified_mimic$$number --finetune False --inference fewshot --topn top$$ranks --save_name biomistral7b_avigon_modified_mimic$$number --max_new_token 300 --prompt_name modified > ./logs/biomistral7b_avigon_modified.log 2>&1; \
		done ;\
		echo "finished for $$number";\
	done

	python3 -c "from src.utils import * ; send_line_message('finished biomistral modified mimic all cases');"



case_study :
# 	top3,5,10
ifneq (,$(wildcard ./logs/casestudy_output.log))
	rm ./logs/casestudy_output.log
	rm -r ./outputs
endif

	for number in 3 5 ; do \
		python3 src/case_study_model_output.py --model mistral7b --finetune False --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model mistral7b --finetune False --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		# python3 src/case_study_model_output.py --model llama3 --finetune False --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		# python3 src/case_study_model_output.py --model llama3 --finetune False --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model llama31 --finetune False --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model llama31 --finetune False --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		echo "finished vanilla";\
		python3 src/case_study_model_output.py --model mistral7b --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model mistral7b --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model mistral7b_mimic --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model mistral7b_mimic --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model biomistral7b_avigon_mimic --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model biomistral7b_avigon_mimic --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		# python3 src/case_study_model_output.py --model llama3 --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		# python3 src/case_study_model_output.py --model llama3 --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		# python3 src/case_study_model_output.py --model llama3_mimic --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		# python3 src/case_study_model_output.py --model llama3_mimic --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model llama31 --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model llama31 --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model llama31_mimic --finetune True --inference zeroshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		python3 src/case_study_model_output.py --model llama31_mimic --finetune True --inference fewshot --topn top$$number  --max_new_token 200 >> ./logs/casestudy_output.log 2>&1; \
		echo "finished finetuned";\
		echo "finished $$number";\
	done

	python3 -c "from src.utils import * ; send_line_message('finished case studies');"

reformat_case_study :
# 	top3,5,10
	ifeq (,"$(wildcard $(./logs/case_study.logs))")
		rm ./logs/case_study.logs
	else 
		echo "starting new case_study.logs"
	endif

	for number in 3 5 10 ; do \
		python3 src/case_study_model_reformat_output.py --model mistral7b --finetune False --inference zeroshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		python3 src/case_study_model_reformat_output.py --model mistral7b --finetune False --inference fewshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		python3 src/case_study_model_reformat_output.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		python3 src/case_study_model_reformat_output.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		echo "finished vanilla";\
		python3 src/case_study_model_reformat_output.py --model mistral7b --finetune True --inference zeroshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		python3 src/case_study_model_reformat_output.py --model mistral7b --finetune True --inference fewshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		python3 src/case_study_model_reformat_output.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		python3 src/case_study_model_reformat_output.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top$$number  --max_new_token 400 >> ./logs/casestudy_output.log 2>>&1; \
		echo "finished finetuned";\
		echo "finished $$number";\
	done

test :
	python3 -c "from src.utils import * ; send_line_message('testing');"
