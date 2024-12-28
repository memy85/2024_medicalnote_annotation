
CUDA_DEVICE_ORDER := PCI_BUS_ID
CUDA_LAUNCH_BLOCKING := 1
CUDA_VISIBLE_DEVICES := 5,6
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
	for number in 10 100 1000 ; do \
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

#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- Following the protocol
# ifeq ($(shell expr $step \< 10 ),1) 
# 	$(eval $step = 10)
# endif

define finetune_mimic
$(eval $MODEL_NAME = $(1))
$(eval $cnt = $(2))
$(eval $step = $(shell expr $(cnt) / 128))
$(if $(shell expr $step \< 10), $(eval $step=10))


python3 src/finetune_models.py \
	--model_name_or_path ${$MODEL_NAME}\
	--data_path data/processed/discharge_dataset_${$cnt}.json \
	--bf16 False \
	--use_fast_tokenizer False \
	--output_dir ./models/${$MODEL_NAME}_mimic_${$cnt} \
	--log_level debug \
	--logging_dir ./logs/models/${$MODEL_NAME}_mimic/logs \
	--num_train_epochs 70 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 128 \
	--model_max_length 20000 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps ${$step} \
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
	> logs/finetune_${$MODEL_NAME}_on_mimic.log

endef

define finetune
$(eval $MODEL_NAME = $(1))

python3 src/finetune_for_each_cv.py --model ${$MODEL_NAME} > ./logs/${$MODEL_NAME}_finetune.log 2>&1
endef 

define protocol_finetune
$(eval $MODEL_NAME = $(1))
echo "----------- FINETUNING WITH ${$MODEL_NAME} ------------"

$(foreach shot, zeroshot fewshot,\
	$(foreach topn, top3 top5 top10, \
		$(call finetune, $($MODEL_NAME), $(shot), $(topn), general);\
		$(call finetune, $($MODEL_NAME), $(shot), $(topn), modified);\
	)\
)

endef

define protocol_cv
$(eval $MODEL_NAME = $(1))
$(eval $FINETUNE = $(2))

echo "----------- CV WITH ${$MODEL_NAME} ------------"
$(foreach shot, zeroshot fewshot, \
	$(foreach topn, top3 top5 top10, \
		$(foreach prompt_name, "modified" "", \
			echo "Starting $(shot), $(topn) $(prompt_name) with ${$MODEL_NAME} + ${$FINETUNE}"
			python3 src/cross_validation.py --model $($MODEL_NAME) --inference $(shot) --topn $(topn) --finetune $($FINETUNE) --just_evaluation True --prompt_name $(prompt_name); \
			python3 -c "from src.utils import * ; send_line_message('finished $(shot) $(topn) $(prompt_name) with ${$MODEL_NAME}');"
		)\
	)\
)
endef

define protocol_test
$(eval $MODEL_NAME = $(1))

echo "----------- TEST WITH ${$MODEL_NAME} ------------"

for shot in zershot fewshot; do \
	echo $$shot ;\
done;

endef

define protocol_increment
$(eval $MODEL_NAME = $(1))

$(foreach cnt, 1000 10000, $(call finetune_mimic, $($MODEL_NAME), $(cnt)))
endef



#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- Fine-Tune

finetune_mistral7b : 
	$(call finetune, mistral7b)

finetune_biomistral7b_avigon : 
	$(call finetune, biomistral7b_avigon)

finetune_llama31 : 
	$(call finetune, llama31)

#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- Cross-Validation

cv_protocol : cv_mistral7b cv_mistral7b_finetuned cv_biomistral7b_avigon cv_biomistral7b_avigon_finetuned cv_llama31 cv_llama31_finetuned

cv_mistral7b : 
	$(call protocol_cv, mistral7b, False)

cv_mistral7b_finetuned : models/mistral7b_finetuned_cv0/checkpoint-50 models/mistral7b_finetuned_cv1/checkpoint-50 models/mistral7b_finetuned_cv2/checkpoint-50 models/mistral7b_finetuned_cv3/checkpoint-50 models/mistral7b_finetuned_cv4/checkpoint-50
	$(call protocol_cv, mistral7b, True)

cv_biomistral7b_avigon :
	$(call protocol_cv, biomistral7b_avigon, False)

cv_biomistral7b_avigon_finetuned : models/biomistral7b_avigon_finetuned_cv0/checkpoint-50 models/biomistral7b_avigon_finetuned_cv1/checkpoint-50 models/biomistral7b_avigon_finetuned_cv2/checkpoint-50 models/biomistral7b_avigon_finetuned_cv3/checkpoint-50 models/biomistral7b_avigon_finetuned_cv4/checkpoint-50
	$(call protocol_cv, biomistral7b_avigon, True)

cv_llama31 :
	$(call protocol_cv, llama31, False)

cv_llama31_finetuned : models/llama31_finetuned_cv0/checkpoint-50 models/llama31_finetuned_cv1/checkpoint-50 models/llama31_finetuned_cv2/checkpoint-50 models/llama31_finetuned_cv3/checkpoint-50 models/llama31_finetuned_cv4/checkpoint-50
	$(call protocol_cv, llama31, True)

#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- baseline

baseline_cv :
	# python3 src/test_baseline.py --model gpt --finetune True --topn top3 --save_name gpt --max_new_token 200 > ./logs/gpt_top3.log 2>&1
	# python3 src/test_baseline.py --model gpt --finetune True --topn top5 --save_name gpt --max_new_token 200 > ./logs/gpt_top5.log 2>&1
	# python3 src/test_baseline.py --model gpt --finetune True --topn top10 --save_name gpt --max_new_token 400 > ./logs/gpt_top10.log 2>&1

	python3 src/test_baseline.py --model biogpt --finetune True --topn top3 --save_name biogpt --max_new_token 200 > ./logs/biogpt_top3.log 2>&1
	python3 src/test_baseline.py --model biogpt --finetune True --topn top5 --save_name biogpt --max_new_token 200 > ./logs/biogpt_top5.log 2>&1
	python3 src/test_baseline.py --model biogpt --finetune True --topn top10 --save_name biogpt --max_new_token 400 > ./logs/biogpt_top10.log 2>&1


#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- increment finetuning

increment_mistral7b :
	$(call protocol_increment, mistral7b)

increment_biomistral7b_avigon :
	$(call protocol_increment, biomistral7b_avigon)

increment_llama31 :
	$(call protocol_increment, llama31)

#################### ----------------------------------------------- ####################
#################### --------------------------------------------------- increment cv

cv_increment_mistral7b : 
	$(foreach cnt, 10 100 1000 10000, $(call protocol_cv, mistral7b_mimic_${cnt}, True))

cv_increment_llama31 : 
	$(foreach cnt, 10 100 1000 10000, $(call protocol_cv, llama31_mimic_${cnt}, True))

# =========================================================================================================================
# --------------------------------------------------- Case Study ----------------------------------------------------------
# =========================================================================================================================

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
