
CUDA_DEVICE_ORDER := PCI_BUS_ID
CUDA_VISIBLE_DEVICES := 0,2,3
# TORCH_EXTENSIONS_DIR :=/home/zhichaoyang/.cache/torch_extensions

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
# export TORCH_EXTENSIONS_DIR

mistal_results: src/test_mistral.py

	# top3
	python3 src/test_mistral.py --model 7b --inference zeroshot --topn top3 --save_name mistral_top3_zeroshot --max_new_token 100 2> logs/mistral_top3_zeroshot.log
	python3 src/test_mistral.py --model 7b --inference fewshot --topn top3 --save_name mistral_top3_fewshot --max_new_token 100 2> logs/mistral_top3_fewshot.log

	# top5
	python3 src/test_mistral.py --model 7b --inference zeroshot --topn top5 --save_name mistral_top5_zeroshot --max_new_token 200  2> logs/mistral_top5_zeroshot.log
	python3 src/test_mistral.py --model 7b --inference fewshot --topn top5 --save_name mistral_top5_fewshot --max_new_token 200    2> logs/mistral_top5_fewshot.log

	# top10
	python3 src/test_mistral.py --model 7b --inference zeroshot --topn top10 --save_name mistral_top10_zeroshot --max_new_token 400   2> logs/mistral_top10_zeroshot.log 
	python3 src/test_mistral.py --model 7b --inference fewshot --topn top10 --save_name mistral_top10_fewshot --max_new_token 400     2> logs/mistral_top10_fewshot.log

finetuned_results: biomistral biollama


biomistral : 
	# top3
	python3 src/test_biollama2_biomistral.py --model biomistral --inference zeroshot --topn top3 --save_name biomistral_top3_zeroshot --max_new_token 200 2> logs/biomistral_top3_zeroshot.log
	python3 src/test_biollama2_biomistral.py --model biomistral --inference fewshot --topn top3 --save_name biomistral_top3_fewshot --max_new_token 200 2> logs/biomistral_top3_fewshot.log

	# top5
	python3 src/test_biollama2_biomistral.py --model biomistral --inference zeroshot --topn top5 --save_name biomistral_top5_zeroshot --max_new_token 200  2> logs/biomistral_top5_zeroshot.log
	python3 src/test_biollama2_biomistral.py --model biomistral --inference fewshot --topn top5 --save_name biomistral_top5_fewshot --max_new_token 200    2> logs/biomistral_top5_fewshot.log

	# top10
	python3 src/test_biollama2_biomistral.py --model biomistral --inference zeroshot --topn top10 --save_name biomistral_top10_zeroshot --max_new_token 400   2> logs/biomistral_top10_zeroshot.log 
	python3 src/test_biollama2_biomistral.py --model biomistral --inference fewshot --topn top10 --save_name biomistral_top10_fewshot --max_new_token 400     2> logs/biomistral_top10_fewshot.log


biollama : 
	# top3
	python3 src/test_biollama2_biomistral.py --model biollama --inference zeroshot --topn top3 --save_name biollama_top3_zeroshot --max_new_token 200 2> logs/biollama_top3_zeroshot.log
	python3 src/test_biollama2_biomistral.py --model biollama --inference fewshot --topn top3 --save_name biollama_top3_fewshot --max_new_token 200 2> logs/biollama_top3_fewshot.log

	# top5
	python3 src/test_biollama2_biomistral.py --model biollama --inference zeroshot --topn top5 --save_name biollama_top5_zeroshot --max_new_token 200  2> logs/biollama_top5_zeroshot.log
	python3 src/test_biollama2_biomistral.py --model biollama --inference fewshot --topn top5 --save_name biollama_top5_fewshot --max_new_token 200    2> logs/biollama_top5_fewshot.log

	# top10
	python3 src/test_biollama2_biomistral.py --model biollama --inference zeroshot --topn top10 --save_name biollama_top10_zeroshot --max_new_token 400   2> logs/biollama_top10_zeroshot.log 
	python3 src/test_biollama2_biomistral.py --model biollama --inference fewshot --topn top10 --save_name biollama_top10_fewshot --max_new_token 400     2> logs/biollama_top10_fewshot.log

meditron : 
	# top3
	python3 src/test_biollama2_biomistral.py --model meditron --inference zeroshot --topn top3 --save_name meditron_top3_zeroshot --max_new_token 200 2> logs/meditron_top3_zeroshot.log
	python3 src/test_biollama2_biomistral.py --model meditron --inference fewshot --topn top3 --save_name meditron_top3_fewshot --max_new_token 200 2> logs/meditron_top3_fewshot.log

	# top5
	python3 src/test_biollama2_biomistral.py --model meditron --inference zeroshot --topn top5 --save_name meditron_top5_zeroshot --max_new_token 200  2> logs/meditron_top5_zeroshot.log
	python3 src/test_biollama2_biomistral.py --model meditron --inference fewshot --topn top5 --save_name meditron_top5_fewshot --max_new_token 200    2> logs/meditron_top5_fewshot.log

	# top10
	python3 src/test_biollama2_biomistral.py --model meditron --inference zeroshot --topn top10 --save_name meditron_top10_zeroshot --max_new_token 400   2> logs/meditron_top10_zeroshot.log 
	python3 src/test_biollama2_biomistral.py --model meditron --inference fewshot --topn top10 --save_name meditron_top10_fewshot --max_new_token 400     2> logs/meditron_top10_fewshot.log

mistral_chain_results: src/test_mistral_v2.py

	# top3
	python3 src/test_mistral_v2.py --model 7b --inference zeroshot --topn top3 --save_name mistral_top3_zeroshot_chain --max_new_token 200 2> logs/mistral_top3_zeroshot.log
	python3 src/test_mistral_v2.py --model 7b --inference fewshot --topn top3 --save_name mistral_top3_fewshot_chain --max_new_token 200 2> logs/mistral_top3_fewshot.log

	# top5
	python3 src/test_mistral_v2.py --model 7b --inference zeroshot --topn top5 --save_name mistral_top5_zeroshot_chain --max_new_token 200  2> logs/mistral_top5_zeroshot.log
	python3 src/test_mistral_v2.py --model 7b --inference fewshot --topn top5 --save_name mistral_top5_fewshot_chain --max_new_token 200    2> logs/mistral_top5_fewshot.log

	# top10
	python3 src/test_mistral_v2.py --model 7b --inference zeroshot --topn top10 --save_name mistral_top10_zeroshot_chain --max_new_token 400   2> logs/mistral_top10_zeroshot.log 
	python3 src/test_mistral_v2.py --model 7b --inference fewshot --topn top10 --save_name mistral_top10_fewshot_chain --max_new_token 400     2> logs/mistral_top10_fewshot.log


finetune_mistral : 
	python3 src/finetune_models.py \
	  --model_name_or_path /data/data_user_alpha/public_models/Mistral/Mistral-7B-Instruct-v0.2 \
      --data_path data/processed/discharge_dataset.json \
      --bf16 False \
      --use_fast_tokenizer False \
      --output_dir ./models/mistral_7b_lora_3 \
      --logging_dir ./logs/models/mistral_7b_lora_3/logs \
      --num_train_epochs 5 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 128 \
      --model_max_length 20000 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 10 \
      --logging_steps 1 \
      --save_total_limit 5 \
      --learning_rate 3e-4 \
      --weight_decay 0.0 \
      --warmup_steps 200 \
      --do_lora True \
      --lora_r 64 \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
      --gradient_checkpointing True 