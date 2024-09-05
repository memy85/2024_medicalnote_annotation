
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

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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
    parser.add_argument("--model", type=str, help="choose model")
    parser.add_argument("--finetune", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot', default='zeroshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10', default='top3')
    parser.add_argument("--save_name", type=str, help='name of the file', default='test')
    parser.add_argument("--max_new_token", type=int, help='max token', default=100)
    parser.add_argument("--quantization", type=str, help="choose either 8 or None", default=None)
    parser.add_argument("--just_evaluation", type=str2bool, nargs='?', const=True, default=False, help="whether to do just evaluation")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = parse_arguments()
    model_name = args.model
    file_name = args.model # the save name and the model name are going to be the same initially
    inference = args.inference
    do_finetune = args.finetune
    topn = args.topn
    max_new_token = args.max_new_token
    quantization = args.quantization
    just_eval_flag = args.just_evaluation
    rank = int(topn.replace("top", ""))
    print("finetune flag : ", do_finetune, file=sys.stderr)

    if quantization is None : 
        quantization_flag = False
    else :
        quantization_flag = True

    MODEL_PATH = config.model_path(model_name)

    # ** if we are already testing a finetuning model
    if "finetuned" in MODEL_PATH :
        fflag = True
    else : 
        fflag = False

    # ** whether to do finetuning from scratch 
    if do_finetune : 
        file_name = f"{model_name}_finetuned"

    print("The MODEL_PATH is ", MODEL_PATH, file=sys.stderr)

    #%%
    # ** explanation
    # cv5 : the cross validation training set list 
    # top10 dataset : medical key word dataset
    # filtered_notes : the EHR notes 
    cv5, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets.pkl"))

    #%%

    # ================================================================================================================ #
    # =============================================== starting CV ==================================================== #
    # ================================================================================================================ #
    
    template = config.template(topn=topn, inference=inference)
    dataset = Dataset.from_pandas(filtered_notes)

    results = []
    # ** explanation 
    # cv5 : a list of cross-validation testset files, 83 files are testset
    # * original data : 104 = 83 (test) + 21 (train)
    # trainset : each files that belongs to the testset
    #%%
    for idx, trainset in enumerate(cv5) : 

        print(f"starting cv ------------------ {idx} ----------------", file=sys.stderr)
        # filter_dataset : the test set
        testset = dataset.filter(lambda x : x["noteid"] not in trainset)

        # ** processed_dataset(testset) is the test set (83 files)
        testset = testset.map(process_texts, batched=True, \
                                        fn_kwargs={"template": template})

        # ** do finetuning if we have finetune flag
        # these are the train set (21 files)
        dataset_name = f"cv{idx}_rank{rank}"
        #%%
        if do_finetune :
            if not just_eval_flag : 
                # ** Name of the finetuned model
                print("doing finetuning & not just evaluation", file=sys.stderr)
                finetune_model_name = f"{model_name}_finetuned_{dataset_name}"
                
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
                model = trainer.model
                tokenizer = trainer.tokenizer

            elif just_eval_flag :
                print("already finetuned models & doing just evaluation", file=sys.stderr)
                finetune_model_name = f"{model_name}_finetuned_{dataset_name}"
                model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
                # checkpoint_path = PROJECT_PATH.joinpath(f"models/{finetune_model_name}/checkpoint-100")
                model = PeftModel.from_pretrained(model, MODEL_PATH)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        else :
            print("vanilla model & doing not just evaluation", file=sys.stderr)
            vanilla_model_name = f"{model_name}_{dataset_name}"
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            tokenizer.add_special_tokens({
                        "eos_token": DEFAULT_EOS_TOKEN,
                        "bos_token": DEFAULT_BOS_TOKEN,
                        "unk_token": DEFAULT_UNK_TOKEN,
                    })
            tokenizer.pad_token = tokenizer.eos_token


        # =================================================================================================== #
        # ============================================ Test model =========================================== #
        # =================================================================================================== #

        print("====================================== Now starting inference", file=sys.stderr)

        torch.cuda.empty_cache()
        questions = testset['questions']
        outputs = []
        for question in tqdm(questions) :
            input_ids = tokenizer(question, return_tensors='pt').input_ids.cuda()
            output_ids = model.generate(input_ids=input_ids, 
                        max_new_tokens = max_new_token)
            
            arr_output = output_ids.detach().cpu().numpy()
            start_of_generate_index = input_ids.shape[1]
            pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]

            print("the text is : ", pred_output, file=sys.stderr)
            outputs.append(pred_output)
            torch.cuda.empty_cache()


        with open(DATA_PATH.joinpath(f"{file_name}_{dataset_name}.pkl"), 'wb') as f :
            pickle.dump(outputs,f)


        # ===================================================================================================== #
        # ============================================ Evaluation ============================================= #
        # ===================================================================================================== #

        #%% 
        gold_dataset = top10_dataset[~top10_dataset.fileid.isin(trainset)].copy()
        
        # erase this below
        # import re
        # from evaluate_model import preprocess_outputs_of_mistral,calculate_mrr
        # file_name = "mistral7b"
        # with open(DATA_PATH.joinpath(f"{file_name}_{dataset_name}.pkl"), 'rb') as f :
        #     outputs = pickle.load(f)
        # pred_dataset = outputs
        # len(pred_dataset)

        #%%

        gold_dataset = gold_dataset[gold_dataset.ranking < rank + 1].copy()
        pred_dataset = outputs

        testset_notes = filtered_notes[~filtered_notes.noteid.isin(trainset)].copy()

        #%% erase below
        # testset['noteid'] # this is the one generates the outputs

        #%% erase below
        # p = re.compile('^\d+')
        # noteid = 'liver_failure.report37286.txt'
        # gold = gold_dataset[gold_dataset.fileid == noteid][['ranking', 'phrase']].copy()
        # gold['ranking'] = gold['ranking'].apply(lambda x : int(p.findall(str(x))[0]))
        # gold = [tuple(x) for x in gold.to_numpy()]
        # pred = preprocess_outputs_of_mistral(outputs[0], rank)

        #%% erase 5
        p, r, f1, m = em.get_results(gold_dataset, pred_dataset, testset['noteid'], rank) 

        #%%

        if do_finetune : 
            record = {"model" : model_name + "_finetuned", "cv" : idx, "topN" : rank, "shots" : inference, "precision" : p, "recall" : r, "f1" : f1, "mrr" : m}
            results.append(record)

        else :
            record = {"model" : model_name, "cv" : idx, "topN" : rank, "shots" : inference, "precision" : p, "recall" : r, "f1" : f1, "mrr" : m}
            results.append(record)

        # ** delete the models in the memory
        if do_finetune :
            del trainer
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    with open(DATA_PATH.joinpath(f"{topn}_{file_name}_{inference}_evaluation.pkl"), 'wb') as f :
        pickle.dump(results, f)
    
    print(f"{dataset_name} finished!", file=sys.stderr)

