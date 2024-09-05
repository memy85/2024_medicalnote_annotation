
CUDA_DEVICE_ORDER := PCI_BUS_ID
CUDA_VISIBLE_DEVICES := 6,7
# TORCH_EXTENSIONS_DIR :=/home/zhichaoyang/.cache/torch_extensions

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
# export TORCH_EXTENSIONS_DIR

mistral7b : src/test_model.py
	# top3
	python3 src/test_model.py --model mistral7b --inference zeroshot --topn top3 --save_name mistral_top3_zeroshot --max_new_token 100 2> logs/mistral_top3_zeroshot.log
	python3 src/test_model.py --model mistral7b --inference fewshot --topn top3 --save_name mistral_top3_fewshot --max_new_token 100 2> logs/mistral_top3_fewshot.log

	# top5
	python3 src/test_model.py --model mistral7b --inference zeroshot --topn top5 --save_name mistral_top5_zeroshot --max_new_token 200  2> logs/mistral_top5_zeroshot.log
	python3 src/test_model.py --model mistral7b --inference fewshot --topn top5 --save_name mistral_top5_fewshot --max_new_token 200    2> logs/mistral_top5_fewshot.log

	# top10
	python3 src/test_model.py --model mistral7b --inference zeroshot --topn top10 --save_name mistral_top10_zeroshot --max_new_token 400   2> logs/mistral_top10_zeroshot.log 
	python3 src/test_model.py --model mistral7b --inference fewshot --topn top10 --save_name mistral_top10_fewshot --max_new_token 400     2> logs/mistral_top10_fewshot.log

finetuned_results: biomistral biollama


biomistral : 
	# top3
	python3 src/test_model.py --model biomistral7b --inference zeroshot --topn top3 --save_name biomistral_top3_zeroshot --max_new_token 200 2> logs/biomistral_top3_zeroshot.log
	python3 src/test_model.py --model biomistral7b --inference fewshot --topn top3 --save_name biomistral_top3_fewshot --max_new_token 200 2> logs/biomistral_top3_fewshot.log

	# top5
	python3 src/test_model.py --model biomistral7b --inference zeroshot --topn top5 --save_name biomistral_top5_zeroshot --max_new_token 200  2> logs/biomistral_top5_zeroshot.log
	python3 src/test_model.py --model biomistral7b --inference fewshot --topn top5 --save_name biomistral_top5_fewshot --max_new_token 200    2> logs/biomistral_top5_fewshot.log

	# top10
	python3 src/test_model.py --model biomistral7b --inference zeroshot --topn top10 --save_name biomistral_top10_zeroshot --max_new_token 400   2> logs/biomistral_top10_zeroshot.log 
	python3 src/test_model.py --model biomistral7b --inference fewshot --topn top10 --save_name biomistral_top10_fewshot --max_new_token 400     2> logs/biomistral_top10_fewshot.log



biollama : 
	# top3
	python3 src/test_model.py --model biollama27b --inference zeroshot --topn top3 --save_name biollama_top3_zeroshot --max_new_token 200 2> logs/biollama_top3_zeroshot.log
	python3 src/test_model.py --model biollama27b --inference fewshot --topn top3 --save_name biollama_top3_fewshot --max_new_token 200 2> logs/biollama_top3_fewshot.log

	# top5
	python3 src/test_model.py --model biollama27b --inference zeroshot --topn top5 --save_name biollama_top5_zeroshot --max_new_token 200  2> logs/biollama_top5_zeroshot.log
	python3 src/test_model.py --model biollama27b --inference fewshot --topn top5 --save_name biollama_top5_fewshot --max_new_token 200    2> logs/biollama_top5_fewshot.log

	# top10
	python3 src/test_model.py --model biollama27b --inference zeroshot --topn top10 --save_name biollama_top10_zeroshot --max_new_token 400   2> logs/biollama_top10_zeroshot.log 
	python3 src/test_model.py --model biollama27b --inference fewshot --topn top10 --save_name biollama_top10_fewshot --max_new_token 400     2> logs/biollama_top10_fewshot.log

mistral7b_finetuned : 
	# top3
	python3 src/test_model_dev.py --model mistral7b-finetuned --inference zeroshot --topn top3 --save_name mistral7b_finetuned_top3_zeroshot --max_new_token 200 2> logs/mistral7b_finetuned_top3_zeroshot.log
	python3 src/test_model_dev.py --model mistral7b-finetuned --inference fewshot --topn top3 --save_name mistral7b_finetuned_top3_fewshot --max_new_token 200 2> logs/mistral7b_finetuned_top3_fewshot.log

	# top5
	python3 src/test_model_dev.py --model mistral7b-finetuned --inference zeroshot --topn top5 --save_name mistral7b_finetuned_top5_zeroshot --max_new_token 200  2> logs/mistral7b_finetuned_top5_zeroshot.log
	python3 src/test_model_dev.py --model mistral7b-finetuned --inference fewshot --topn top5 --save_name mistral7b_finetuned_top5_fewshot --max_new_token 200    2> logs/mistral7b_finetuned_top5_fewshot.log

	# top10
	python3 src/test_model_dev.py --model mistral7b-finetuned --inference zeroshot --topn top10 --save_name mistral7b_finetuned_top10_zeroshot --max_new_token 400   2> logs/mistral7b_finetuned_top10_zeroshot.log 
	python3 src/test_model_dev.py --model mistral7b-finetuned --inference fewshot --topn top10 --save_name mistral7b_finetuned_top10_fewshot --max_new_token 400     2> logs/mistral7b_finetuned_top10_fewshot.log


biomistral7b_finetuned : 
	# top3
	python3 src/test_model_dev.py --model biomistral7b-finetuned --inference zeroshot --topn top3 --save_name biomistral7b_finetuned_top3_zeroshot --max_new_token 200 2> logs/biomistral7b_finetuned_top3_zeroshot.log
	python3 src/test_model_dev.py --model biomistral7b-finetuned --inference fewshot --topn top3 --save_name biomistral7b_finetuned_top3_fewshot --max_new_token 200 2> logs/biomistral7b_finetuned_top3_fewshot.log

	# top5
	python3 src/test_model_dev.py --model biomistral7b-finetuned --inference zeroshot --topn top5 --save_name biomistral7b_finetuned_top5_zeroshot --max_new_token 200  2> logs/biomistral7b_finetuned_top5_zeroshot.log
	python3 src/test_model_dev.py --model biomistral7b-finetuned --inference fewshot --topn top5 --save_name biomistral7b_finetuned_top5_fewshot --max_new_token 200    2> logs/biomistral7b_finetuned_top5_fewshot.log

	# top10
	python3 src/test_model_dev.py --model biomistral7b-finetuned --inference zeroshot --topn top10 --save_name biomistral7b_finetuned_top10_zeroshot --max_new_token 400   2> logs/biomistral7b_finetuned_top10_zeroshot.log 
	python3 src/test_model_dev.py --model biomistral7b-finetuned --inference fewshot --topn top10 --save_name biomistral7b_finetuned_top10_fewshot --max_new_token 400     2> logs/biomistral7b_finetuned_top10_fewshot.log
	
biomistral7b_avigon :
	# top3
	python3 src/test_model.py --model biomistral7b-avigon --inference zeroshot --topn top3 --save_name biomistral7b_avigon_top3_zeroshot --max_new_token 200 2> logs/biomistral7b_avigon_top3_zeroshot.log
	python3 src/test_model.py --model biomistral7b-avigon --inference fewshot --topn top3 --save_name biomistral7b_avigon_top3_fewshot --max_new_token 200 2> logs/biomistral7b_avigon_top3_fewshot.log

	# top5
	python3 src/test_model.py --model biomistral7b-avigon --inference zeroshot --topn top5 --save_name biomistral7b_avigon_top5_zeroshot --max_new_token 200  2> logs/biomistral7b_avigon_top5_zeroshot.log
	python3 src/test_model.py --model biomistral7b-avigon --inference fewshot --topn top5 --save_name biomistral7b_avigon_top5_fewshot --max_new_token 200    2> logs/biomistral7b_avigon_top5_fewshot.log

	# top10
	python3 src/test_model.py --model biomistral7b-avigon --inference zeroshot --topn top10 --save_name biomistral7b_avigon_top10_zeroshot --max_new_token 400   2> logs/biomistral7b_avigon_top10_zeroshot.log 
	python3 src/test_model.py --model biomistral7b-avigon --inference fewshot --topn top10 --save_name biomistral7b_avigon_top10_fewshot --max_new_token 400     2> logs/biomistral7b_avigon_top10_fewshot.log

biomistral7b_avigon_finetuned : 
	# top3
	python3 src/test_model_dev.py --model biomistral7b-avigon-finetuned --inference zeroshot --topn top3 --save_name biomistral7b_avigon_finetuned_top3_zeroshot --max_new_token 200 2> logs/biomistral7b_avigon_finetuned_top3_zeroshot.log
	python3 src/test_model_dev.py --model biomistral7b-avigon-finetuned --inference fewshot --topn top3 --save_name biomistral7b_avigon_finetuned_top3_fewshot --max_new_token 200 2> logs/biomistral7b_avigon_finetuned_top3_fewshot.log

	# top5
	python3 src/test_model_dev.py --model biomistral7b-avigon-finetuned --inference zeroshot --topn top5 --save_name biomistral7b_avigon_finetuned_top5_zeroshot --max_new_token 200 2> logs/biomistral7b_avigon_finetuned_top5_zeroshot.log
	python3 src/test_model_dev.py --model biomistral7b-avigon-finetuned --inference fewshot --topn top5 --save_name biomistral7b_avigon_finetuned_top5_fewshot --max_new_token 200 2> logs/biomistral7b_avigon_finetuned_top5_fewshot.log

	# top10
	python3 src/test_model_dev.py --model biomistral7b-avigon-finetuned --inference zeroshot --topn top10 --save_name biomistral7b_avigon_finetuned_top10_zeroshot --max_new_token 400 2> logs/biomistral7b_avigon_finetuned_top10_zeroshot.log 
	python3 src/test_model_dev.py --model biomistral7b-avigon-finetuned --inference fewshot --topn top10 --save_name biomistral7b_avigon_finetuned_top10_fewshot --max_new_token 400 2> logs/biomistral7b_avigon_finetuned_top10_fewshot.log


# evaluate : 
# 	python3 src/evaluate_model.py --model mistral
# 	python3 src/evaluate_model.py --model mistral7b_finetuned
# 	python3 src/evaluate_model.py --model biomistral
# 	python3 src/evaluate_model.py --model biomistral7b_finetuned
# 	python3 src/evaluate_model.py --model biomistral7b_avigon
# 	python3 src/evaluate_model.py --model biomistral7b_avigon_finetuned



finetune_mistral_on_jinying : 
	python3 src/finetune_models.py \
	  --model_name_or_path mistral7b \
      --data_path data/processed/jinying_sample_dataset.json \
      --bf16 False \
      --use_fast_tokenizer False \
      --output_dir ./models/mistral_7b_lora_jinying \
      --logging_dir ./logs/models/mistral_7b_lora_jinying/logs \
      --num_train_epochs 200 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 128 \
      --model_max_length 20000 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 10 \
      --logging_steps 1 \
      --save_total_limit 50 \
      --learning_rate 3e-4 \
      --weight_decay 0.0 \
      --warmup_steps 200 \
      --do_lora True \
      --lora_r 64 \
      --report_to tensorboard \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
      --gradient_checkpointing True \
      --load_in_8bit False \
      > logs/finetune_mistral7b.log


# model_name_or_path should be biomistral7b
finetune_biomistral_on_jinying : 
	python3 src/finetune_models.py \
	  --model_name_or_path biomistral7b \
      --data_path data/processed/jinying_sample_dataset.json \
      --bf16 False \
      --use_fast_tokenizer False \
      --output_dir ./models/biomistral_7b_lora_jinying \
      --logging_dir ./logs/models/biomistral_7b_lora_jinying/logs \
      --num_train_epochs 200 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 128 \
      --model_max_length 20000 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 10 \
      --logging_steps 1 \
      --report_to tensorboard \
      --save_total_limit 50 \
      --learning_rate 3e-4 \
      --weight_decay 0.0 \
      --warmup_steps 200 \
      --do_lora True \
      --lora_r 64 \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
      --gradient_checkpointing True \
      --load_in_8bit False > logs/finetune_biomistral7b.log


finetune_biomistral_avigon_on_jinying : 
	python3 src/finetune_models.py \
	  --model_name_or_path biomistral7b-avigon \
      --data_path data/processed/jinying_sample_dataset.json \
      --bf16 False \
      --use_fast_tokenizer False \
      --output_dir ./models/biomistral_avigon_7b_lora_jinying \
      --log_level debug \
      --logging_dir ./logs/models/biomistral_avigon_7b_lora_jinying/logs \
      --num_train_epochs 300 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 128 \
      --model_max_length 20000 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 10 \
      --logging_steps 1 \
      --save_total_limit 50 \
      --learning_rate 3e-4 \
      --weight_decay 0.0 \
      --warmup_steps 200 \
      --do_lora True \
      --lora_r 64 \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
      --gradient_checkpointing True \
      --load_in_8bit False \
      --report_to tensorboard \
      > logs/finetune_biomistral7b_avigon_on_jinying.log


cv_no_finetune :
#	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top3 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top3_zeroshot.log 2>&1
#	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top3 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top3_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top3 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top3 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top3_fewshot.log 2>&1

#	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top5 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top5_zeroshot.log 2>&1
#	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top5 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_top5_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top5 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top5 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_top5_fewshot.log 2>&1

#	python3 src/cross_validation.py --model mistral7b --finetune False --inference zeroshot --topn top10 --save_name mistral7b --max_new_token 400 > ./logs/mistral7b_top10_zeroshot.log 2>&1
#	python3 src/cross_validation.py --model mistral7b --finetune False --inference fewshot --topn top10 --save_name mistral7b --max_new_token 400 > ./logs/mistral7b_top10_fewshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference zeroshot --topn top10 --save_name biomistral7b_avigon --max_new_token 400 > ./logs/biomistral7b_avigon_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune False --inference fewshot --topn top10 --save_name biomistral7b_avigon --max_new_token 400 > ./logs/biomistral7b_avigon_top10_fewshot.log 2>&1

cv_finetune_rank3 :
	python3 src/cross_validation.py --model mistral7b --finetune True --inference zeroshot --topn top3 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b --finetune True --inference fewshot --topn top3 --save_name mistral7b --max_new_token 200 > ./logs/mistral7b_finetune_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b --finetune True --inference zeroshot --topn top3 --save_name biomistral7b --max_new_token 200 > ./logs/biomistral7b_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b --finetune True --inference fewshot --topn top3 --save_name biomistral7b --max_new_token 200 > ./logs/biomistral7b_finetune_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference zeroshot --topn top3 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_avigon --finetune True --inference fewshot --topn top3 --save_name biomistral7b_avigon --max_new_token 200 > ./logs/biomistral7b_avigon_finetune_top3_fewshot.log 2>&1

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



finetune_mistral_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path mistral7b \
      --data_path data/processed/discharge_dataset_train.json \
	  --eval_data_path data/processed/discharge_dataset_test.json \
      --bf16 False \
      --use_fast_tokenizer False \
      --output_dir ./models/mistral_7b_mimic \
      --logging_dir ./logs/models/mistral_7b_mimic/logs \
      --num_train_epochs 200 \
      --per_device_train_batch_size 1 \
	  --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 128 \
      --model_max_length 20000 \
      --evaluation_strategy "steps" \
      --save_strategy "steps" \
      --save_steps 10 \
      --logging_steps 1 \
      --logging_strategy "steps" \
      --save_total_limit 50 \
      --learning_rate 3e-4 \
      --weight_decay 0.0 \
      --warmup_steps 200 \
      --do_lora True \
      --lora_r 64 \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
      --report_to tensorboard \
      --gradient_checkpointing True \
      > logs/finetune_mistral7b_mimic.log

finetune_biomistral_on_mimic : 
	python3 src/finetune_models.py \
	  --model_name_or_path biomistral7b_avigon \
      --data_path data/processed/discharge_dataset_train.json \
	  --eval_data_path data/processed/discharge_dataset_test.json \
      --bf16 False \
      --use_fast_tokenizer False \
      --output_dir ./models/biomistral_7b_mimic \
      --logging_dir ./logs/models/biomistral_7b_mimic/logs \
      --num_train_epochs 200 \
      --per_device_train_batch_size 1 \
	  --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 128 \
      --model_max_length 20000 \
      --evaluation_strategy "steps" \
      --save_strategy "steps" \
      --save_steps 10 \
      --logging_steps 1 \
      --logging_strategy "steps" \
      --save_total_limit 50 \
      --learning_rate 3e-4 \
      --weight_decay 0.0 \
      --warmup_steps 200 \
      --do_lora True \
      --lora_r 64 \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
      --report_to tensorboard \
      --gradient_checkpointing True \
      > logs/finetune_biomistral7b_mimic.log

cv_mistral_mimic_finetune :
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top3 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top3 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top5 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top5 --save_name mistral7b_mimic --max_new_token 200 > ./logs/mistral7b_mimic_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top10 --save_name mistral7b_mimic --max_new_token 400 > ./logs/mistral7b_mimic_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model mistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top10 --save_name mistral7b_mimic --max_new_token 400 > ./logs/mistral7b_mimic_top10_fewshot.log 2>&1

cv_biomistral_mimic_finetune :
	python3 src/cross_validation.py --model biomistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top3 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top3_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top3 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top3_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top5 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top5_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top5 --save_name biomistral7b_mimic --max_new_token 200 > ./logs/biomistral7b_mimic_top5_fewshot.log 2>&1

	python3 src/cross_validation.py --model biomistral7b_mimic_finetuned --finetune False --just_evaluation True --inference zeroshot --topn top10 --save_name biomistral7b_mimic --max_new_token 400 > ./logs/biomistral7b_mimic_top10_zeroshot.log 2>&1
	python3 src/cross_validation.py --model biomistral7b_mimic_finetuned --finetune False --just_evaluation True --inference fewshot --topn top10 --save_name biomistral7b_mimic --max_new_token 400 > ./logs/biomistral7b_mimic_top10_fewshot.log 2>&1


modified_cv_no_finetune :
	python3 src/modified_cross_validation.py --model mistral7b --finetune False --inference modified_zeroshot --topn top5 --save_name mistral7b_modified --max_new_token 200 > ./logs/mistral7b_top5_zeroshot.log 2>&1
	python3 src/modified_cross_validation.py --model mistral7b --finetune False --inference modified_fewshot --topn top5 --save_name mistral7b_modified --max_new_token 200 > ./logs/mistral7b_top5_fewshot.log 2>&1
	python3 src/modified_cross_validation.py --model biomistral7b_avigon --finetune False --inference modified_zeroshot --topn top5 --save_name biomistral7b_avigon_modified --max_new_token 200 > ./logs/biomistral7b_avigon_top5_zeroshot.log 2>&1
	python3 src/modified_cross_validation.py --model biomistral7b_avigon --finetune False --inference modified_fewshot --topn top5 --save_name biomistral7b_avigon_modified --max_new_token 200 > ./logs/biomistral7b_avigon_top5_fewshot.log 2>&1

#	python3 src/modified_cross_validation.py --model mistral7b --finetune False --inference modified_zeroshot --topn top10 --save_name mistral7b_modified --max_new_token 500 > ./logs/mistral7b_top10_zeroshot.log 2>&1
#	python3 src/modified_cross_validation.py --model mistral7b --finetune False --inference modified_fewshot --topn top10 --save_name mistral7b_modified --max_new_token 500 > ./logs/mistral7b_top10_fewshot.log 2>&1
#	python3 src/modified_cross_validation.py --model biomistral7b_avigon --finetune False --inference modified_zeroshot --topn top10 --save_name biomistral7b_avigon_modified --max_new_token 500 > ./logs/biomistral7b_avigon_top10_zeroshot.log 2>&1
#	python3 src/modified_cross_validation.py --model biomistral7b_avigon --finetune False --inference modified_fewshot --topn top10 --save_name biomistral7b_avigon_modified --max_new_token 500 > ./logs/biomistral7b_avigon_top10_fewshot.log 2>&1