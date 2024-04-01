
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
PROMPT_PATH = PROJECT_PATH.joinpath("prompts")


def process_texts(samples, template) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = template.format(
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose between biollama or biomistral")
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10')
    parser.add_argument("--save_name", type=str, help='name of the file')
    parser.add_argument("--max_new_token", type=int, help='max token', default=100)
    parser.add_argument("--quantization", type=str, help="choose either 8 or None", default=None)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = parse_arguments()
    inference = args.inference
    topn = args.topn
    file_name = args.save_name
    model = args.model
    max_new_token = args.max_new_token
    quantization = args.quantization

    if quantization is None : 
        quantization_flag = False
    else :
        quantization_flag = True
    
    if model == "biollama" :
        MODEL_PATH = config.model_path('biollama27b')
    elif model == 'biomistral' : 
        MODEL_PATH = config.model_path('biomistral7b')
    else :
        MODEL_PATH = config.model_path('meditron')


    top3_dataset, top5_dataset, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("processed_ranking_datasets.pkl"))

    template = config.template(topn, inference)

    # merge the notes
    dataset = Dataset.from_pandas(filtered_notes)

    processed_dataset = dataset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": template})
    
    print(processed_dataset['questions'][0], file=sys.stderr)

    pipe = pipeline('text-generation', 
                    MODEL_PATH, 
                    device_map='auto',
                    model_kwargs={"load_in_8bit":quantization_flag})


    print("====================================== Now starting inference", file=sys.stderr)

    torch.cuda.empty_cache()
    sample = processed_dataset['questions']
    outputs = []
    for text in tqdm(sample) :
        output = pipe(text, 
            return_full_text=False, 
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            max_new_tokens=max_new_token)

        print("the text is : ", output, file=sys.stderr)
        outputs.append(output)
        torch.cuda.empty_cache()

    outputs = list(map(lambda x : x[0]['generated_text'], outputs))

    import pickle
    with open(DATA_PATH.joinpath(f"{file_name}.pkl"), 'wb') as f :
        pickle.dump(outputs,f)

#%%
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from datasets import Dataset
# import pandas as pd
# import torch
# import torch.nn as nn
# import sys
# import argparse
# from tqdm import tqdm
# from utils import *


# config = load_config()
# PROJECT_PATH = config.project_path
# DATA_PATH = PROJECT_PATH.joinpath("data/processed")
# PROMPT_PATH = PROJECT_PATH.joinpath("prompts")
# MODEL_PATH = config.model_path('meditron')

# #%%
# pipe = pipeline('text-generation', 
#                 MODEL_PATH, 
#                 device_map='cpu')
# #%%
# output = pipe("introduce yourself", 
#     return_full_text=False, 
#     do_sample=True,
#     top_p=0.95,
#     max_new_tokens=100)

# #%%

