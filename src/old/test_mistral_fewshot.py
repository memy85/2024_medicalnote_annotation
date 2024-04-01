
#%%
import os
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import Dataset
import pandas as pd
from peft import PeftModel
import torch
import torch.nn as nn
import os
from utils import *
config = load_config()


torch.cuda.empty_cache()
num_devices = torch.cuda.device_count()
print("number of devices : ", num_devices)
for idx in range(num_devices) : 
    print("device is :", torch.cuda.get_device_name(idx))



MISTRAL_PATH = '/data/data_user_alpha/public_models/Mistral-8x7B-Instruct-v0.1'
#%%
template = config.template('instructiontune')
instructions = config.template('instructions')

print("there a ~ e styles of instructions.\nThe instruction tells the model to extract important terms.\n")
print(instructions)

mergedData = pd.read_pickle("../data/processed/mergedData.pkl")
#%%
# ========================================================= Build dataset and do modeling part
# lets build a dataset 

dataset = Dataset.from_pandas(mergedData)

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN= "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
max_target_length = 10000


#%%
config = load_config()
instruction = config.template('instructions')
i1 = instruction['a']
i2 = instruction['hieu']
i3 = instruction['fewshot']

fewshot_template = config.template('fewshot')

#%%

def process_texts(samples) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = fewshot_template.format(
            instruction = i3,
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}

processed_dataset = dataset.map(process_texts, batched=True)

#%%
# from transformers import pipeline
pipe = pipeline('text-generation', 
                MISTRAL_PATH, 
                # tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map='auto')


#%%

set_seed(42)
torch.cuda.empty_cache()
sample = processed_dataset['questions']
outputs = []
for text in sample :
    output = pipe(text, 
         return_full_text=False, 
         do_sample=True,
         top_p=0.95,
         max_new_tokens=512)

    print("the text is : ", output)
    outputs.append(output)
    torch.cuda.empty_cache()

#%%
import pickle
with open("../data/processed/fewshot_mistral.pkl", 'wb') as f :
    pickle.dump(outputs,f)

#%%
print("====================================== Now starting inference")
