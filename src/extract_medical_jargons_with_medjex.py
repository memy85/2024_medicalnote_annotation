#%%
import os

os.environ["CUDA_LAUNCH_BLOCKING"]="1" 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import pickle

from tqdm import tqdm
import pandas as pd
import torch
import sqlite3
from utils import *

config = load_config()
from medjex import MedJEx, BINARY_model, MAX_TOKEN_LEN, tokenizer


PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath('data')


device = "cuda" if torch.cuda.is_available() else "cpu"
medjex_model = MedJEx(BINARY_model, device)

# discharge_path = DATA_PATH.joinpath('raw/discharge.csv')

con = sqlite3.connect(DATA_PATH.joinpath('raw/medicalnote.db'))
discharge_path = DATA_PATH.joinpath('raw/discharge.csv')
raw_data = pd.read_sql_query("select * from discharge", con=con)

#%%
# mednlp = medspacy.load()
# idx = 7
# output = raw_data['text'][idx]
# print(output)

# mytexts = [sent.text for sent in mednlp(output).sents if sent.text]
# mytexts_length = list(map(lambda x : len(x) ,mytexts))
# max(mytexts_length)

#%%
# MAX_TOKEN_LEN = 512
# def break_down_to_max_length(text, max_token_length = MAX_TOKEN_LEN) :
#     if len(text) > max_token_length :
#         segment_front = text[:max_token_length]
#         segment_back = text[max_token_length:]
#         print(len(segment_front))

#         # find the first space in front segment
#         flag = True
#         idx = -1
#         while flag :
#             if segment_front[idx] != ' ' :
#                 idx -= 1
#                 continue
#             else :
#                 segment_back = segment_front[idx:] + segment_back
#                 segment_front = segment_front[:idx]
#                 break

#         segment_front = segment_front.strip()
#         segment_back = segment_back.strip()
#         return [segment_front] + break_down_to_max_length(segment_back, max_token_length=MAX_TOKEN_LEN)
#     else : # basecase
#         return [text]
# #%%
# out = break_down_to_max_length(sample)
# medjex_model.predict(output)

#%%

jargon_list = []
for idx, row in tqdm(raw_data.iterrows()) : 
    output = medjex_model.predict(row['text'])
    jargon_list.append(output)

raw_data['extracted_jargons'] = jargon_list

#%%
with open(DATA_PATH.joinpath('processed/discharge_extracted_jargons_v2.pkl'), 'wb') as f :
    pickle.dump(raw_data, f)
