#%%
import os
import pickle
import gc
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel

from peft.config import PeftConfig
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
import sys
import argparse
from tqdm import tqdm
from utils import *

import finetune_models
import evaluate_model as em


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts")

#%%
def process_texts(samples, template) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = template.format(
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}
#%%

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot', default='zeroshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10', default='top3')
    parser.add_argument("--save_name", type=str, help='name of the file', default='test')
    parser.add_argument("--max_new_token", type=int, help='max token', default=200)
    parser.add_argument("--quantization", type=str, help="choose either 8 or None", default=None)
    parser.add_argument("--baseline", type=str2bool, nargs='?', const=True, default=False, help="whether this is a baseline model or not")
    parser.add_argument("--prompt_name", type=str, help="name of prompt", default=None)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = parse_arguments()
    model_name = args.model
    # file_name is the name that is the exact name of the model. whether it's finetuned or mimic finetuned
    # inference = args.inference
    # topn = args.topn
    max_new_token = args.max_new_token
    quantization = args.quantization
    save_name = args.save_name
    baseline_flag = args.baseline
    # prompt_name = args.prompt_name

    # rank = int(topn.replace("top", ""))

    if quantization is None : 
        quantization_flag = False
    else :
        quantization_flag = True

    # ** whether to do finetuning from scratch 
    # real_name = f"{model_name}_finetuned"

    # print("The real name is ", real_name)

    #%%
    # ** explanation
    # cv5 : the cross validation training set list 
    # top10 dataset : medical key word dataset
    # filtered_notes : the EHR notes 
    # cv5, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_new.pkl"))
    cv10, top10_dataset, filtered_notes = load_cv_dataset()

    #%%

    # ================================================================================================================ #
    # =============================================== starting CV ==================================================== #
    # ================================================================================================================ #
    
    # template = config.template(topn=topn, inference=inference, prompt_name=prompt_name)
    dataset = Dataset.from_pandas(filtered_notes)

    results = []
    # ** explanation 
    # cv5 : a list of cross-validation testset files, 83 files are testset
    # * original data : 104 = 83 (test) + 21 (train)
    # trainset : each files that belongs to the testset
    #%%
    for cv_idx, (train_idx, test_idx) in enumerate(cv10) : 
        # if idx < 4 : 
        #     continue
        print(f"----------- ------------------ {idx} ----------------", file=sys.stderr)
        print(f"starting cv ------------------ {idx} ----------------", file=sys.stderr)
        print(f"----------- ------------------ {idx} ----------------", file=sys.stderr)

        # ** do finetuning if we have finetune flag
        # these are the train set (21 files)
        dataset_name = f"cv{cv_idx}"

        # ** Name of the finetuned model
        print("doing finetuning", file=sys.stderr)
        finetune_model_name = f"{model_name}_finetuned_{dataset_name}"
        MODEL_PATH = config.model_path(model_name=model_name)
        
        # --lora_weight_path ./models/{finetune_model_name}
        arguments = f'''
        --model_name_or_path {model_name}
        --data_path data/processed/finetune/{dataset_name}.json
        --bf16 False 
        --use_fast_tokenizer False 
        --output_dir ./models/{finetune_model_name} 
        --logging_dir ./logs/models/{finetune_model_name}
        --num_train_epochs 50
        --per_device_train_batch_size 1 
        --gradient_accumulation_steps 128 
        --model_max_length 20000 
        --save_strategy steps 
        --save_steps 10 
        --logging_steps 1 
        --save_total_limit 50 
        --learning_rate 3e-4 
        --weight_decay 0.0 
        --warmup_steps 200 
        --do_lora True 
        --lora_r 64 
        --report_to tensorboard 
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj" 
        --gradient_checkpointing True 
        --load_in_8bit False 
        '''

        trainer = finetune_models.train(arguments)

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        send_line_message(f"for {finetune_model_name}, the finetuning is done")


