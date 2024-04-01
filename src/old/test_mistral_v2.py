
#%%
import os
import logging
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
import sys
import argparse
from tqdm import tqdm
from utils import *


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
max_target_length = 50


def process_texts(samples, template, instruction) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = template.format(
            instruction = instruction,
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose between 8b and 87b")
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot')
    parser.add_argument("--instruction", type=str, help='choose between default top3, top5')
    parser.add_argument("--save_name", type=str, help='name of the file')
    parser.add_argument("--quantization", type=str, help="choose either 8 or None", default=None)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = parse_arguments()
    file_name = args.save_name
    model = args.model
    quantization = args.quantization

    if quantization is None : 
        quantization_flag = False
    else :
        quantization_flag = True

    MISTRAL_PATH = config.model_path(f'mistral{model}')

    top3_dataset, top5_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("processed_ranking_datasets.pkl"))



    instruction = config.template('instructions')
    template = config.template(args.inference)
    # choose instruction type
    i1 = instruction[args.instruction]

    # merge the notes
    dataset = Dataset.from_pandas(filtered_notes)

    processed_dataset = dataset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": template, "instruction" : i1})
    
    print(processed_dataset['questions'][0], file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(MISTRAL_PATH)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_PATH)
    tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    tokenizer.pad_token = tokenizer.eos_token


    print("====================================== Succesfully loaded the model", file=sys.stderr)
    print("====================================== Now starting inference", file=sys.stderr)

    torch.cuda.empty_cache()
    sample = processed_dataset['questions']
    outputs = []
    for text in tqdm(sample) :
        input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
        output_ids = model.generate(input_ids=input_ids, 
                       max_new_tokens = max_target_length)
        
        arr_output = output_ids.detach().cpu().numpy()
        start_of_generate_index = input_ids.shape[1]
        pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]

        print("the text is : ", pred_output, file=sys.stderr)
        outputs.append(pred_output)
        torch.cuda.empty_cache()


    import pickle
    with open(DATA_PATH.joinpath(f"{file_name}.pkl"), 'wb') as f :
        pickle.dump(outputs,f)


