
#%%
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "backend:native,max_split_size_mb:256"
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
import pandas as pd
import torch.nn as nn
import torch
import pickle
from utils import *

config = load_config()

# MISTRAL_PATH = '/data/data_user_alpha/public_models/Mistral-8x7B-Instruct-v0.1'
MISTRAL_PATH = '/data/data_user_alpha/public_models/Mistral/Mistral-7B-Instruct-v0.2'

mergedData = pd.read_pickle("../data/processed/mergedData.pkl")

# ========================================================= Build dataset and do modeling part
# lets build a dataset 

dataset = Dataset.from_pandas(mergedData)

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN= "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
max_target_length = 1000


tokenizer = AutoTokenizer.from_pretrained(MISTRAL_PATH, cache_dir=None)
tokenizer.add_special_tokens({
    'eos_token' : DEFAULT_EOS_TOKEN,
    'bos_token' : DEFAULT_BOS_TOKEN,
    'unk_token' : DEFAULT_UNK_TOKEN,
})

config = load_config()
instruction = config.template('instructions')
i1 = instruction['b']
i2 = instruction['hieu']

tokenizer.pad_token = tokenizer.eos_token

zeroshot_template = config.template('zeroshot')

def process_texts(samples) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = zeroshot_template.format(
            instruction = i1,
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}

processed_dataset = dataset.map(process_texts, batched=True)

# tokenize the texts
model = AutoModelForCausalLM.from_pretrained(MISTRAL_PATH, 
                                             device_map="auto")

print("============================== Successfully Loaded the Model =========================================")

tokenized_texts = tokenizer(processed_dataset['questions'], 
                            return_tensors='pt', 
                            max_length=8000,
                            truncation='only_first',
                            padding=True).to("cuda")


input_ids = tokenized_texts.input_ids


#%%
# generated_ids = model.generate(input_ids = input_ids,
#                     max_new_tokens=100, 
#                     do_sample=True, 
#                     top_p=0.95)

# arr_output = generated_ids.detach().cpu().numpy()
# start_of_generate_index = tokenized_texts.input_ids.shape[1]
# pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)

#%%

print("====================================== Now starting inference")
# print("using cpu inference=====================")

outputs = []
for x in input_ids : 
    start = time.time()
    out = model.generate(input_ids = x.reshape(1,-1), 
                        max_new_tokens=100, 
                        do_sample=True, 
                        top_p=0.95)
    outputs.append(out.detach().cpu().numpy())
    del x
    end = time.time()
    ("the total time is ", end - start)



#%%

arr_output = out.detach().cpu().numpy()
# now decode the outputs
start_of_generate_index = tokenized_texts.input_ids.shape[1]
pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)
#%%
print(pred_output[0])

#%%
# 
# with open('llama2_output.pkl', 'wb') as f :
#     pickle.dump(pred_output, f)
