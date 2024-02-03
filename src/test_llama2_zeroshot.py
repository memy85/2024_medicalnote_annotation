
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pickle
from datasets import Dataset
import pandas as pd
from peft import PeftModel
import torch
import torch.nn as nn
import os
from utils import *
config = load_config()


# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

LLAMA_PATH = "/home/htran/generation/biomed_instruct/models/llama_7b_lora/checkpoint-970"
LLAMA2_PATH = "/home/htran/generation/biomed_instruct/models/llama_2_7b_all_instructions/checkpoint-975"


#%%
template = config.template('instructiontune')
instructions = config.template('instructions')

print("there a ~ e styles of instructions.\nThe instruction tells the model to extract important terms.\n")
print(instructions)


annotation_info_table = pd.read_pickle("../data/processed/annotation_info_table.pkl")
filtered_notes = pd.read_pickle("../data/processed/filtered_notes.pkl")


def get_note(noteid) :
    text = filtered_notes[filtered_notes['noteid'] == noteid]['text']
    annotations = annotation_info_table[annotation_info_table['noteid'] == noteid]['concept']
    annotations = annotations.tolist()
    return text, annotations



note_sample = filtered_notes.loc[2]['text']
# print(note_sample)



annotation_info_table['concept_modified'] = annotation_info_table.concept.apply(lambda x : str(x) + ", ")
annotations = annotation_info_table.groupby('noteid')['concept_modified'].sum()



# merge the notes
merged_notes = filtered_notes.merge(annotations, on='noteid')
merged_notes['concept_modified'] = merged_notes['concept_modified'].str.strip()
# merged_notes.head(15)

#%%
# ========================================================= Build dataset and do modeling part

# lets build a dataset 

dataset = Dataset.from_pandas(merged_notes)
# dataset

# first test llama & llama2
# model = pipeline("text-generation", LLAMA2_PATH, max_new_tokens=128, device_map="auto")
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES=1,2,3

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN= "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
# max_target_length = 4096
max_target_length = 1000




#%%
model = AutoModelForCausalLM.from_pretrained(LLAMA2_PATH, device_map="balanced")
model = PeftModel.from_pretrained(model, 
                                  LLAMA2_PATH)

print("loaded models")

#%%
tokenizer = AutoTokenizer.from_pretrained(LLAMA2_PATH, cache_dir=None)
tokenizer.add_special_tokens({
    'eos_token' : DEFAULT_EOS_TOKEN,
    'bos_token' : DEFAULT_BOS_TOKEN,
    'unk_token' : DEFAULT_UNK_TOKEN,
})

config = load_config()
instruction = config.template('instructions')
i1 = instruction['a']
i2 = instruction['hieu']

tokenizer.pad_token = tokenizer.eos_token

zeroshot_template = config.template('zeroshot')

#%%

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
tokenized_texts = tokenizer(processed_dataset['questions'], 
                            return_tensors='pt', 
                            max_length=1000,
                            truncation='only_first',
                            padding=True)


#%%

# from transformers import pipeline
# pipe = pipeline('text-generation', 
#                 LLAMA2_PATH, 
#                 tokenizer=tokenizer, 
#                 device_map='sequential')

#%%

# sample = processed_dataset['questions']
# output = pipe(sample, 
#      return_full_text=False, 
#      do_sample=True, 
#      batch_size=2,
#      max_new_tokens=512,)


#%%
print("====================================== Now starting inference")

input_ids = tokenized_texts.input_ids
print("allocated data")

#%%

# generation
# outputs = []
# for idx, ids in enumerate(input_ids) : 
#     output = model.generate(input_ids = input_ids.reshape(1,-1), 
#                         max_new_tokens = 50)
#     outputs.append(output)
#     print("done idx : ", idx)
    
#     break


#%%
x = input_ids[0]
output = model.generate(input_ids = x.reshape(-1,1), max_new_tokens = 100)



#%%

arr_output = output.detach().cpu().numpy()
# now decode the outputs
start_of_generate_index = tokenized_texts.input_ids.shape[1]
pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)

# %%
