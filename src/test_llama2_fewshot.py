import os, sys
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"

import pandas as pd
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from utils import *

# first test llama & llama2
# model = pipeline("text-generation", LLAMA2_PATH, max_new_tokens=128, device_map="auto")
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

config = load_config()

LLAMA2_PATH = "/home/htran/generation/biomed_instruct/models/llama_2_7b_all_instructions/checkpoint-975"
PROJECT_PATH = Path(config.project_path)

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN= "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
max_target_length = 4096

model = AutoModelForCausalLM.from_pretrained(LLAMA2_PATH, device_map='sequential',cache_dir=None)

tokenizer = AutoTokenizer.from_pretrained(LLAMA2_PATH, cache_dir=None)
tokenizer.add_special_tokens({
    'eos_token' : DEFAULT_EOS_TOKEN,
    'bos_token' : DEFAULT_BOS_TOKEN,
    'unk_token' : DEFAULT_UNK_TOKEN,
})

tokenizer.pad_token = tokenizer.eos_token

instruction = config.template('instructions')
i1 = instruction['a']
i2 = instruction['hieu']

fewshot_template = config.template('fewshot')




merged_notes = pd.read_pickle(PROJECT_PATH.joinpath("data/processed/mergedData.pkl"))


def process_texts(samples) :

    example = config.template("fewshot_example")
    texts = samples['text']
    formated_texts = []
    for text in texts :
        contexts = example.format(instruction = i1)
        new_text = fewshot_template.format(
            instruction = i1,
            context = text
        )
        formated_texts.append(new_text)
    
    return {"contexts": [contexts]*len(formated_texts), "questions" : formated_texts}


dataset = Dataset.from_pandas(merged_notes)

processed_dataset = dataset.map(process_texts, batched=True)


# tokenize the texts
tokenized_texts = tokenizer(processed_dataset['contexts'], 
                            processed_dataset['questions'], 
                            return_tensors='pt', 
                            padding=True,
                            max_length=4000,
                            truncation='only_first').to('cuda')


with torch.no_grad() :
    output = model.generate(**tokenized_texts, max_new_tokens = 500)



# now decode the outputs
start_of_generate_index = tokenized_texts.input_ids.shape[1]
pred_output = tokenizer.batch_decode(output[:, start_of_generate_index:], skip_special_tokens=True)

with open(PROJECT_PATH.joinpath("data/processed/fewshot_output.pkl"), 'wb') as f :
    pickle.dump(pred_output, f)