
#%%
import os
import logging
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


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts")

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def process_texts(samples, template) :
    """
    for a given samples, extracts the 'text' and attatches it to the 
    template. 
    input : samples, template
    """

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
    parser.add_argument("--model", type=str, help="choose model")
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot', default='zeroshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10', default='top3')
    parser.add_argument("--save_name", type=str, help='name of the file', default='test')
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


    MODEL_PATH = config.model_path(model)
    if "finetuned" in MODEL_PATH :
        fflag = True
    else : 
        fflag = False

    print("The MODEL_PATH is ", MODEL_PATH, file=sys.stderr)

    top3_dataset, top5_dataset, top10_dataset, filtered_notes, _, _ = pd.read_pickle(DATA_PATH.joinpath("processed_ranking_datasets.pkl"))

    template = config.template(topn, inference)

    # merge the notes
    dataset = Dataset.from_pandas(filtered_notes)

    processed_dataset = dataset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": template})
    
    print(processed_dataset['questions'][0], file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    tokenizer.pad_token = tokenizer.eos_token

    if fflag : 
        config = PeftConfig.from_pretrained(MODEL_PATH)
        model = PeftModel.from_pretrained(model, MODEL_PATH)

    print("====================================== Now starting inference", file=sys.stderr)

    torch.cuda.empty_cache()
    sample = processed_dataset['questions']
    outputs = []
    for text in tqdm(sample) :
        input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
        output_ids = model.generate(input_ids=input_ids, 
                       max_new_tokens = max_new_token)
        
        arr_output = output_ids.detach().cpu().numpy()
        start_of_generate_index = input_ids.shape[1]
        pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]

        print("the text is : ", pred_output, file=sys.stderr)
        outputs.append(pred_output)
        torch.cuda.empty_cache()
        # break


    # print(outputs, file=sys.stderr)
    # outputs = list(map(lambda x : x[0]['generated_text'], outputs))
    # print(outputs, file=sys.stderr)

    import pickle
    with open(DATA_PATH.joinpath(f"{file_name}.pkl"), 'wb') as f :
        pickle.dump(outputs,f)
