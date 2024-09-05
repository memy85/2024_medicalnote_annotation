'''
Description 
- this code extracts the jargons using the quickUMLS extraction
'''
#%%
import pickle
from collections import defaultdict, Counter
from functools import reduce
import pandas as pd
import numpy as np
from tqdm import tqdm
from quickumls import get_quickumls_client

from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
n_samples = 10000

discharge_dataset_path = DATA_PATH.joinpath("mimic_discharge.pkl")
discharge_dataset = pd.read_pickle(discharge_dataset_path)
matcher = get_quickumls_client()

#%%
def filter_out_terms(terms, overlap_terms) :
    terms = list(reduce(lambda x, y : x + y, terms))
    filtered = list(filter(lambda x : x['similarity'] >= 1.0, terms))
    words = []
    for term in filtered :
        word = term['ngram']
        if word in list(overlap_terms) :
            words.append(word)
    return words



def extract_overlapping_terms(front_jargons, back_jargons) :
    f = list(reduce(lambda x, y : x + y, front_jargons))
    b = list(reduce(lambda x, y : x + y, back_jargons))
    f = list(map(lambda x : x['term'], f))
    b = list(map(lambda x : x['term'], b))
    f, b = set(f), set(b)
    overlap = f & b
    return overlap


front_terms = Counter()
back_terms = Counter()
number_of_terms_appear_on_corpus = Counter()
N = len(discharge_dataset)
number_of_terms_appear_on_docs = []
counts = 0
filtered_discharge_dataset = []
for idx, row in tqdm(discharge_dataset.iterrows()) : 
    
    if counts == n_samples :
        break

    # Initialize Counter for this document
    c = Counter()
    # Extract the texts, front and back(discharge summary)
    front = row['front_text']
    back = row['discharge_instructions']

    # perform matching for each front and backs
    front_jargons = matcher.match(front, best_match=True, ignore_syntax=False)
    back_jargons = matcher.match(back, best_match=True, ignore_syntax=False)
    
    if len(front_jargons) == 0 or len(back_jargons) == 0 :
        continue

    # 1. find overlapping terms
    overlapping_terms = extract_overlapping_terms(front_jargons, back_jargons)
    if len(overlapping_terms) < 10 :
        continue
    # overlapping_term_books.append(overlapping_terms) # this is a set
    
    # 2. filter out terms with overlapping terms. These are list
    fronts = filter_out_terms(front_jargons, overlap_terms=overlapping_terms)
    backs = filter_out_terms(back_jargons, overlap_terms=overlapping_terms)
    
    # extend list and count for this document
    all_filtered_terms = fronts + backs
    for term in all_filtered_terms :
        c[term] += 1

    # 3. add the term counts to the number_of_terms_appear_on_corpus
    for term in list(overlapping_terms) :
        number_of_terms_appear_on_corpus[term] += 1
    
    # 4. append the c to overlapping_term_books
    number_of_terms_appear_on_docs.append(c)

    # 5. append used row
    filtered_discharge_dataset.append(row)

    # 6. count 
    counts += 1

filtered_discharge_dataset = pd.DataFrame(filtered_discharge_dataset)

# Now we calculate the TF-IDF
tfidf_list = []
for i in range(len(number_of_terms_appear_on_docs)) :

    # Initialize counter
    tf = Counter()
    tfidf = Counter()

    # calculate tf
    for term, count in number_of_terms_appear_on_docs[i].items() :
        tf[term] = count / sum(number_of_terms_appear_on_docs[i].values())
    
    # calculate tf-idf
    for term, count in number_of_terms_appear_on_docs[i].items() :
        tfidf[term] = tf[term] * (np.log(N/number_of_terms_appear_on_corpus[term]))
    
    tfidf_list.append(tfidf)
    


# discharge_dataset.iloc[2]['front_jargon'] = fronts
# discharge_dataset.iloc[2]['back_jargon'] = backs
# with open(DATA_PATH.joinpath("front_jargons.pkl"),'wb') as f :
#     pickle.dump(fronts, f)

# with open(DATA_PATH.joinpath("back_jargons.pkl"),'wb') as f :
#     pickle.dump(backs, f)
#%%

# Save Tf-IDF 
with open(DATA_PATH.joinpath("tfidf_jargons.pkl"),'wb') as f :
    pickle.dump(tfidf_list, f)

with open(DATA_PATH.joinpath("filtered_discharge_dataset.pkl"), "wb") as f :
    pickle.dump(filtered_discharge_dataset, f)

print("finished!")
