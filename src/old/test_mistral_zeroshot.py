
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
from utils import *


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MISTRAL_PATH = config.model_path('mistral')


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
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot')
    parser.add_argument("--instruction", type=str, help='choose between default top3, top5')
    parser.add_argument("--save_name", type=str, help='name of the file')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = parse_arguments()
    file_name = args.save_name

    mergedData = pd.read_pickle("../data/processed/mergedData.pkl")

    instruction = config.template('instructions')
    template = config.template(args.inference)
    i1 = instruction[args.instruction]

    # merge the notes
    dataset = Dataset.from_pandas(mergedData)

    processed_dataset = dataset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": args.inference, "instruction" : i1})

    pipe = pipeline('text-generation', 
                    MISTRAL_PATH, 
                    torch_dtype=torch.bfloat16,
                    device_map='auto')


    print("====================================== Now starting inference", file=sys.stderr)

    sample = processed_dataset['questions']
    output = pipe(sample, 
        return_full_text=False, 
        do_sample=True, 
        max_new_tokens=256)


    import pickle
    with open(DATA_PATH.joinpath(f"{file_name}.pkl"), 'wb') as f :
        pickle.dump(output,f)


