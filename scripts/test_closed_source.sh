#!/bin/bash

for MODEL in 'gpt-4o-mini' 'gpt-4o' ; do 
	# python3 src/test_closed_source.py --model $MODEL --provider openai
	for SHOT in 'zeroshot' 'fewshot'; do
		for PROMPT in 'generic' 'structured' ; do 
			python3 src/aggregate_scores.py --model $MODEL --finetune false --shot $SHOT  --prompt_name $PROMPT
		done
	done
done

# for MODEL in 'claude-3-haiku-20240307' 'claude-sonnet-4-20250514' 'claude-opus-4-1-20250805' ; do 
# 	python3 src/test_closed_source.py --model $MODEL --provider anthropic
#
# 	for SHOT in 'zeroshot' 'fewshot'; do
# 		for PROMPT in 'generic' 'structured' ; do 
# 			python3 src/aggregate_scores.py --model $MODEL --finetune false --shot $SHOT  --prompt_name $PROMPT
# done
