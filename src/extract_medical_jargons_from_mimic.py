

#%%
import pickle
import pandas as pd
from tqdm import tqdm
from quickumls import get_quickumls_client

from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

discharge_dataset_path = DATA_PATH.joinpath("mimic_discharge.pkl")
discharge_dataset = pd.read_pickle(discharge_dataset_path)
matcher = get_quickumls_client()

def extract_terms(terms) :
    filtered = list(filter(lambda x : x['similarity'] >= 1.0, terms))
    words = []
    for term in filtered : 
        word = term['ngram']
        words.append(word)
    return set(words)

#%%
fronts = []
backs = []
for idx, row in tqdm(discharge_dataset.iterrows()) : 

    # sample = discharge_dataset['discharge_instructions'][0]
    front = row['front_text']
    back = row['discharge_instructions']

    front_jargons = matcher.match(front, best_match=True, ignore_syntax=False)
    back_jargons = matcher.match(back, best_match=True, ignore_syntax=False)

    myset = set()
    for f in front_jargons :
        output = extract_terms(f)
        myset = myset | output

    front_jargons = myset

    myset = set()
    for b in back_jargons :
        output = extract_terms(b)
        myset = myset | output

    back_jargons = myset
    fronts.append(front_jargons)
    backs.append(back_jargons)

    if idx == 10000 :
        break
    

# discharge_dataset.iloc[2]['front_jargon'] = fronts
# discharge_dataset.iloc[2]['back_jargon'] = backs
#%%
with open(DATA_PATH.joinpath("front_jargons.pkl"),'wb') as f :
    pickle.dump(fronts, f)

with open(DATA_PATH.joinpath("back_jargons.pkl"),'wb') as f :
    pickle.dump(backs, f)

# discharge_dataset.to_pickle(DATA_PATH.joinpath("jargons_extracted.pkl"))
print("finished!")
