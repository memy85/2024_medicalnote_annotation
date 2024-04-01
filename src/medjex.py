
import random
import os
from os.path import exists

from pprint import pprint as pprint
import pickle
import json
from typing import List, Optional
from copy import copy
import urllib.request

import numpy as np
import pandas
import sklearn

import torch
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig

from tqdm import tqdm

from MedJEx.loader.loader import load_file, load_data, load_ner_file
from MedJEx.utils.sequence_labeler import SequenceLabeler
from MedJEx.utils.scorers import F1
from MedJEx.utils.MedCAT import MedCAT_wrapper
from MedJEx.utils.normalization import Normalizer
from MedJEx.utils.term_weighting import TermFrequency, MLM_weight

from MedJEx.models.models import EarlyAndLateFusionBertMLPCRF, EarlyAndLateFusionrobertaMLPCRF
from nltk.tokenize import word_tokenize

import medspacy

manual_seed = 0
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)

### ===================================================== ###
### ---> ** set this part accustomed to your path ** <--- ###
### ===================================================== ###

UMLS_MedCAT_PATH = '/home/wjang/medicalnote_annotation/src/MedJEx/umls_large.zip'
DEFAULT_PATH = '/home/wjang/medicalnote_annotation/src/MedJEx'

UMLS_matcher = MedCAT_wrapper(UMLS_MedCAT_PATH)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = os.path.join(DEFAULT_PATH, 'data')
NOTE_AID_PATH = os.path.join(DATA_PATH, 'sample.csv')

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

wiki = True
normalization = True
normalization_type = 'min_max'
MAX_TOKEN_LEN = 512
Binary_flag = False
TF_flag = False
MLM_flag = False
additional_feature = False
batch_size = 1

# if not os.path.isfile('MedJEx.pth'):
#     urllib.request.urlretrieve("https://huggingface.co/Mozzi/MedJEx/resolve/main/model.pth", "results/MedJEx.pth")

BINARY_PATH = os.path.join(DEFAULT_PATH, 'results/MedJEx.pth')
PROCESSED_DICT_FILE = os.path.join(DEFAULT_PATH, 'data/%s_processed_data_dict.pkl'%MODEL_NAME)
PROCESSED_SPLIT_DICT_FILE = os.path.join(DEFAULT_PATH, 'data/%s_data_split_dict.pkl'%MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

note_aid_data_dict = load_file(NOTE_AID_PATH)
BIOES_labeler = SequenceLabeler(labeling_scheme='BIOES',longest_labeling_flag=True)
UMLS_labeler = SequenceLabeler(labeling_scheme='BIOES',longest_labeling_flag=True)


if os.path.exists(PROCESSED_DICT_FILE):
    processed_data_dict = pickle.load(open(PROCESSED_DICT_FILE,'rb'))

else:
    processed_data_dict = load_data(note_aid_data_dict, tokenizer, BIOES_labeler, UMLS_matcher = UMLS_matcher, UMLS_labeler=UMLS_labeler)
    with open(PROCESSED_DICT_FILE,'wb') as f :
        pickle.dump(processed_data_dict, f)

# print(processed_data_dict[list(processed_data_dict.keys())[0]])

data_splits = {}
if os.path.exists(PROCESSED_SPLIT_DICT_FILE):
    data_splits = pickle.load(open(PROCESSED_SPLIT_DICT_FILE,'rb'))
    print("processed dict file already exists!")
    
else:
    data_splits = {}
    train_data_portion = 1.0
    keys = list(processed_data_dict.keys())
    random.shuffle(keys)
    
    train_keys = keys[:int(0.8*len(keys))]
    dev_keys = keys[int(0.8*len(keys)):int(0.90*len(keys))]
    test_keys = keys[int(0.9*len(keys)):]
    
    train_split = [processed_data_dict[key] for key in train_keys if len(processed_data_dict[key]['token_ids']) < MAX_TOKEN_LEN]
    train_split = train_split[:int(len(train_split)*train_data_portion)]
    dev_split = [processed_data_dict[key] for key in dev_keys if len(processed_data_dict[key]['token_ids'])  < MAX_TOKEN_LEN]
    test_split = [processed_data_dict[key] for key in test_keys if len(processed_data_dict[key]['token_ids'])  < MAX_TOKEN_LEN]
    
    TF_weighting = TermFrequency()
    train_split = TF_weighting.get_weights(processed_data=train_split, target_entities='entities')
    dev_split = TF_weighting.get_weights(processed_data=dev_split, target_entities='entities')
    test_split = TF_weighting.get_weights(processed_data=test_split, target_entities='entities')
    train_split = TF_weighting.get_weights(processed_data=train_split, target_entities='UMLS_concepts')
    dev_split = TF_weighting.get_weights(processed_data=dev_split, target_entities='UMLS_concepts')
    test_split = TF_weighting.get_weights(processed_data=test_split, target_entities='UMLS_concepts')
    
    MLM_weighting = MLM_weight(MODEL_NAME)
    train_split = MLM_weighting.get_weights(processed_data=train_split, target_entities='entities')
    dev_split = MLM_weighting.get_weights(processed_data=dev_split, target_entities='entities')
    test_split = MLM_weighting.get_weights(processed_data=test_split, target_entities='entities')
    train_split = MLM_weighting.get_weights(processed_data=train_split, target_entities='UMLS_concepts')
    dev_split = MLM_weighting.get_weights(processed_data=dev_split, target_entities='UMLS_concepts')
    test_split = MLM_weighting.get_weights(processed_data=test_split, target_entities='UMLS_concepts')
    
    data_splits['train_split'] = train_split
    data_splits['dev_split'] = dev_split
    data_splits['test_split'] = test_split
    
    pickle.dump(data_splits, open(PROCESSED_SPLIT_DICT_FILE,'wb'))
    
    train_keys = [data['sentid'] for data in train_split]
    test_keys = [data['sentid'] for data in test_split]
    dev_keys = [data['sentid'] for data in dev_split]

    data_split = {'train_sents': train_keys,
                  'test_sents': test_keys,
                  'dev_sents': dev_keys}

    str_data_split = json.dumps(data_split)

    SPLIT_DATA_PATH = os.path.join(DEFAULT_PATH,"train_dev_test_sent_nums.jsonl")
    fin = open(SPLIT_DATA_PATH,'w')
    fin.write(str_data_split)
    fin.close()


train_split = data_splits['train_split']
dev_split = data_splits['dev_split']
test_split = data_splits['test_split'] 


def data_normalization(data_splits, normalizer, target_entities):
    for data_index, data_split in enumerate(data_splits):
        concepts = data_split[target_entities]
        for concept_index, concept in enumerate(concepts):
            data_splits[data_index][target_entities][concept_index]['term_frequency'] = normalizer.normalizer['term_frequency'].get_normalized_results(concept['term_frequency'])
            data_splits[data_index][target_entities][concept_index]['MLM_weight'] = normalizer.normalizer['MLM_weight'].get_normalized_results(concept['MLM_weight'])
    return data_splits


if normalization:
    normalizer = Normalizer(train_split, normalization_type = normalization_type)
    
    train_split = data_normalization(train_split, normalizer, target_entities = 'UMLS_concepts')
    dev_split = data_normalization(dev_split, normalizer, target_entities = 'UMLS_concepts')
    test_split = data_normalization(test_split, normalizer, target_entities = 'UMLS_concepts')
    
    train_split = data_normalization(train_split, normalizer, target_entities = 'entities')
    dev_split = data_normalization(dev_split, normalizer, target_entities = 'entities')
    test_split = data_normalization(test_split, normalizer, target_entities = 'entities')


class JargonTerm(Dataset):

    def __init__(self, JargonTerm_data, tokenizer, labeler, MAX_TOKEN_LEN=256, Binary_flag = False, TF_flag = False, MLM_flag = False, UMLS_lablers = None):
        self.data = JargonTerm_data
        self.pad_id = tokenizer.pad_token_id
        self.outter_id = labeler.label2id['O']
        self.MAX_TOKEN_LEN = MAX_TOKEN_LEN
        
        self.TF_flag = TF_flag
        self.MLM_flag = MLM_flag
        self.Binary_flag = Binary_flag
        
        self.additional_feature_flag = False

        if Binary_flag or TF_flag or MLM_flag:
            self.additional_feature_flag = True

        #     self.num_of_additional_features = len(UMLS_lablers.label2id)
        # else:
        #     self.additional_feature_flag = False
        #     self.num_of_additional_features = 0
        
        self.num_of_additional_features = 0
        self.num_of_binary_features = 0
        self.num_of_weighted_features = 0

        self.flags = [self.TF_flag, self.MLM_flag]
        
        if True in self.flags:
            self.num_of_additional_features += len(UMLS_lablers.label2id)
        self.num_of_additional_features += sum([len(UMLS_lablers.label2id) if flag else 0 for flag in self.flags])
        
        if self.Binary_flag:
            self.num_of_binary_features += len(UMLS_lablers.label2id)
        if True in self.flags:
            self.num_of_weighted_features += sum([len(UMLS_lablers.label2id) if flag else 0 for flag in self.flags])
        
        #self.num_of_UMLS_labels = len(UMLS_lablers.label2id)
        
    def __getitem__(self, idx):

        def _padding(loaded_data, MAX_TOKEN_LEN):
            token_len = len(loaded_data['token_ids'])
            margin_len = self.MAX_TOKEN_LEN - token_len
            
            input_ids = loaded_data['token_ids'] + [self.pad_id] * margin_len
            label_ids = loaded_data['yids'] + [self.outter_id] * margin_len
            attention_mask = [1] * token_len + [0] * margin_len
            token_type_ids = [0] * token_len + [0] * margin_len
            
            return input_ids, attention_mask, token_type_ids, label_ids

        
        def _truncating(loaded_data, MAX_TOKEN_LEN):
            input_ids = loaded_data['token_ids'][:MAX_TOKEN_LEN]
            label_ids = loaded_data['yids'][:MAX_TOKEN_LEN] 
            attention_mask = [1] * MAX_TOKEN_LEN
            token_type_ids = [0] * MAX_TOKEN_LEN
            
            return input_ids, attention_mask, token_type_ids, label_ids

        
        def _term_features(loaded_data, MAX_TOKEN_LEN):
            token_len = len(loaded_data['token_ids'])
            margin_len = self.MAX_TOKEN_LEN - token_len
            
            concepts = loaded_data['UMLS_concepts']

            # UMLS_yids = loaded_data['UMLS_yids']
            
            # UMLS_bin_representation = loaded_data['UMLS_bin_representation']
            # UMLS_bin_dim = UMLS_bin_representation.shape[1]
            # UMLS_bin_representation_margin = np.zeros((margin_len, UMLS_bin_dim))
            # UMLS_bin_representation = np.concatenate((UMLS_bin_representation, UMLS_bin_representation_margin), axis = 0)

            def binary_feature_map(concepts, tokens, feature=None):
                labelings = []
                for index, concept in enumerate(concepts):
                    # set as a first concept among the candidates
                    #concept = concept[0]
                    cui = concept['cui']; term = concept['term']; semtypes=concept['semtypes']
                    start_token = concept['start_token']; end_token = concept['end_token'];

                    if feature:
                        weight = concept[feature]
                    else:
                        weight = 1.0
                    
                    labelings.append(((start_token, end_token, cui), weight))
                
                UMLS_bin_dim = len(UMLS_labeler.id2label)
                token_len = len(tokens)
                UMLS_bin_representation = np.zeros((self.MAX_TOKEN_LEN, UMLS_bin_dim))
                
                for token_index, representation in enumerate(UMLS_bin_representation):
                    UMLS_bin_representation[token_index][UMLS_labeler.label2id['O']] = 1.0
                
                for labeling in labelings:
                    weight = labeling[1]; labeling = labeling[0];
                    UMLS_ylabels, UMLS_yids = UMLS_labeler.get_labels(tokens, [labeling])
                    for token_id, yid in enumerate(UMLS_yids):
                            if UMLS_labeler.label2id['O'] != yid and token_id < 512:
                                UMLS_bin_representation[token_id][yid] = 1.0 #weight
                                UMLS_bin_representation[token_id][UMLS_labeler.label2id['O']] = 0.0
                
                return UMLS_bin_representation

            
            def weighted_feature_map(concepts, tokens, feature=None):
                labelings = []
                for index, concept in enumerate(concepts):
                    # set as a first concept among the candidates
                    #concept = concept[0]
                    cui = concept['cui']; term = concept['term']; semtypes=concept['semtypes']
                    start_token = concept['start_token']; end_token = concept['end_token'];

                    if feature:
                        weight = concept[feature]
                    else:
                        weight = 1.0
                    
                    labelings.append(((start_token, end_token, cui), weight))
                
                UMLS_bin_dim = len(UMLS_labeler.id2label)
                token_len = len(tokens)
                UMLS_weighted_representation = np.zeros((self.MAX_TOKEN_LEN, UMLS_bin_dim))
                
                # for token_index, representation in enumerate(UMLS_bin_representation):
                #     UMLS_bin_representation[token_index][UMLS_labeler.label2id['O']] = 1.0
                
                for labeling in labelings:
                    weight = labeling[1]; labeling = labeling[0];
                    UMLS_ylabels, UMLS_yids = UMLS_labeler.get_labels(tokens, [labeling])
                    for token_id, yid in enumerate(UMLS_yids):
                            if UMLS_labeler.label2id['O'] != yid:
                                UMLS_weighted_representation[token_id][yid] = weight
                                UMLS_weighted_representation[token_id][UMLS_labeler.label2id['O']] = 0.0
                
                return UMLS_weighted_representation
                
            
            additional_features = {}
            if self.TF_flag or self.MLM_flag or self.Binary_flag:
                bin_weights = binary_feature_map(concepts, loaded_data['token_ids'], feature = 'term_frequency')
                additional_features['bin_weights'] = torch.tensor(bin_weights, dtype = torch.float32)
            if self.TF_flag:
                    TF_weights = weighted_feature_map(concepts, loaded_data['token_ids'], feature = 'term_frequency')
                    additional_features['TF_weights'] = torch.tensor(TF_weights, dtype = torch.float32)
            if self.MLM_flag:
                    MLM_weights = weighted_feature_map(concepts, loaded_data['token_ids'], feature = 'MLM_weight')
                    additional_features['MLM_weights'] = torch.tensor(MLM_weights, dtype = torch.float32)

#             additional_features = {}
            
#             # initialize addiatinal feature lists
#             # UMLS_flags = np.zeros((MAX_TOKEN_LEN, self.num_of_UMLS_labels))
            
#             if self.TF_flag:
#                 TF_weights = np.zeros((MAX_TOKEN_LEN, 1))
#             if self.MLM_flag:
#                 MLM_weights = np.zeros((MAX_TOKEN_LEN, 1))
                
            
#             for concept in concepts:
#                 start_token = concept['start_token']; end_token = concept['end_token']
#                 MLM_weight = concept['MLM_weight']; term_frequency = concept['term_frequency']
                
#                 for token_index in range(start_token, end_token):
#                     # UMLS_flags[token_index] = 1 
#                     if self.TF_flag:
#                         TF_weights[token_index][0] = .0 #term_frequency 
#                     if self.MLM_flag:
#                         MLM_weights[token_index][0] =  .0 #MLM_weight
#             # for token_index, UMLS_yid in enumerate(UMLS_yids):
#             #     UMLS_flags[token_index][UMLS_yid] = 1.0
                    
#             # additional_features['UMLS_flags'] = torch.tensor(UMLS_flags, dtype = torch.float32)
#             additional_features['UMLS_flags'] = torch.tensor(UMLS_bin_representation, dtype = torch.float32)
            
            return additional_features
        
        loaded_data = self.data[idx]

        if len(loaded_data['token_ids']) < self.MAX_TOKEN_LEN:
            processed_data = _padding(loaded_data, self.MAX_TOKEN_LEN)

        else:
            processed_data = _truncating(loaded_data, self.MAX_TOKEN_LEN)

        input_ids, attention_mask, token_type_ids, labels = processed_data
        
        
        item = {}
        item['input_ids'] = torch.tensor(input_ids);
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = torch.tensor(token_type_ids)
        item['labels'] = torch.tensor(labels)
        item['sentid'] = loaded_data['sentid']
        
        if self.additional_feature_flag:
            additional_features = _term_features(loaded_data, self.MAX_TOKEN_LEN)
            #item['UMLS_flags'] = additional_features['UMLS_flags']
            item['bin_weights'] = additional_features['bin_weights']
            if self.TF_flag:
                item['TF_weights'] = additional_features['TF_weights']
            if self.MLM_flag:
                item['MLM_weights'] = additional_features['MLM_weights']
            
        else: item['additional_features'] = {}
        
        return item

    def __len__(self):
        return len(self.data)


class evaluation_model(object):
    
    def __init__(self, PATH, data_splits, tokenizer, BIOES_labeler, UMLS_labeler):
        model_data = torch.load(PATH)
        self.model_data  = model_data
        
        train_split = data_splits['train_split']
        dev_split = data_splits['dev_split']
        test_split = data_splits['test_split'] 
        
        normalization = model_data['normalization']
        normalization_type = model_data['normalization_type']
        # normalization =True
        # normalization_type = 'min_max'
        if normalization:
            
            normalizer = Normalizer(train_split, normalization_type = normalization_type)
            train_split = data_normalization(train_split, normalizer, target_entities = 'UMLS_concepts')
            dev_split = data_normalization(dev_split, normalizer, target_entities = 'UMLS_concepts')
            test_split = data_normalization(test_split, normalizer, target_entities = 'UMLS_concepts')
            
            
        MAX_TOKEN_LEN = 128
        
        Binary_flag = model_data['Binary_flag']
        TF_flag = model_data['TF_flag']
        MLM_flag = model_data['MLM_flag']
        additional_feature = model_data['additional_feature']
        
        train_dataset = JargonTerm(train_split, tokenizer, BIOES_labeler, MAX_TOKEN_LEN, Binary_flag, TF_flag, MLM_flag, UMLS_labeler)
        dev_dataset = JargonTerm(dev_split, tokenizer, BIOES_labeler, MAX_TOKEN_LEN, Binary_flag, TF_flag, MLM_flag, UMLS_labeler)
        test_dataset = JargonTerm(test_split, tokenizer, BIOES_labeler, MAX_TOKEN_LEN, Binary_flag, TF_flag, MLM_flag, UMLS_labeler)
        
        num_of_additional_features = {}
        num_of_additional_features['num_of_binary_features'] = train_dataset.num_of_binary_features
        num_of_additional_features['num_of_weighted_features'] = train_dataset.num_of_weighted_features

        if not additional_feature:
            num_of_additional_features = None
        else:
            if not TF_flag and not MLM_flag:
                num_of_additional_features['num_of_weighted_features'] = 0
        
        batch_size = model_data['batch_size']
        
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size)
        dev_loader = DataLoader(dev_dataset, batch_size = batch_size)
        
        pretrained_model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=BIOES_labeler.num_of_label)
        # model = BERT_MLP.from_pretrained(MODEL_NAME, config=pretrained_model_config, num_of_additional_features = num_of_additional_features)
        if 'roberta' in  MODEL_NAME:
            self.model = EarlyAndLateFusionrobertaMLPCRF.from_pretrained(MODEL_NAME, config=pretrained_model_config, num_of_additional_features = num_of_additional_features)
        else:
            self.model = EarlyAndLateFusionBertMLPCRF.from_pretrained(MODEL_NAME, config=pretrained_model_config, num_of_additional_features = num_of_additional_features)
        
        
        self.model.load_state_dict(model_data['model_state_dict'], strict=False)
        
        self.Binary_flag = Binary_flag
        self.TF_flag = TF_flag
        self.MLM_flag = MLM_flag
        self.additional_feature = additional_feature

        self.dev_split = dev_split
        self.test_split = test_split
        
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        
        self.batch_size = batch_size


class MedJEx(object):

    def __init__(self, BINARY_model, device):
        self.BINARY_model = BINARY_model
        
        self.model = BINARY_model.model.to(device)
        self.dev_loader = BINARY_model.dev_loader
        
        self.MLM_weighting = MLM_weight(MODEL_NAME)
        self.TF_weighting = TermFrequency()
        self.mednlp = medspacy.load()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
     
    def break_down_to_max_length(self, text, max_token_length = MAX_TOKEN_LEN) :
        if len(text) > max_token_length :
            segment_front = text[:max_token_length]
            segment_back = text[max_token_length:]

            # find the first space in front segment
            flag = True
            idx = -1
            while flag :
                if segment_front[idx] != ' ' :
                    idx -= 1
                    continue
                else :
                    segment_back = segment_front[idx:] + segment_back
                    segment_front = segment_front[:idx]
                    break

            segment_front = segment_front.strip()
            segment_back = segment_back.strip()
            return [segment_front] + self.break_down_to_max_length(segment_back, max_token_length=MAX_TOKEN_LEN)

        else : # basecase
            print("finished processing!")
            return [text]

    def predict(self, input_text, bz = 4, ignore_MLM_weighting = False):
        
        # in case the sentence length > 512 or max length,
        # we parse it to handle these. 

        def _sentencify(mednlp, text):

            chunks = self.break_down_to_max_length(text, max_token_length=MAX_TOKEN_LEN)
            sents = [sent.text for chunk in chunks for sent in mednlp(chunk).sents if sent.text ]
            # sents = [sent.text for sent in mednlp(text).sents if sent.text]
            # parsed_sents = []
            # for sent in sents :
                # parsed_sents.extend(self.break_down_to_max_length(sent))

            # sents = [sent.text.strip() for sent in mednlp(text).sents if sent.text]
            # return parsed_sents
            return sents

        torch.cuda.empty_cache()
        sents = _sentencify(self.mednlp, input_text)
        note_aid_data_dict = load_sents(sents)
        processed_data_dict = load_data(note_aid_data_dict, 
                                        self.tokenizer, 
                                        BIOES_labeler, 
                                        UMLS_matcher = UMLS_matcher, 
                                        UMLS_labeler=UMLS_labeler)

        keys = list(processed_data_dict.keys())
        test_keys = keys

        test_split = [processed_data_dict[key] for key in test_keys if len(processed_data_dict[key]['token_ids'])]
        
        TF_weighting = self.TF_weighting
        test_split = TF_weighting.get_weights(processed_data=test_split, target_entities='entities')
        test_split = TF_weighting.get_weights(processed_data=test_split, target_entities='UMLS_concepts')

        MLM_weighting = self.MLM_weighting
        test_split = MLM_weighting.get_weights(processed_data=test_split, target_entities='entities', ignore = ignore_MLM_weighting)
        test_split = MLM_weighting.get_weights(processed_data=test_split, target_entities='UMLS_concepts', ignore = ignore_MLM_weighting)


        #temp_model = BINARY_model
        model = self.model
        dev_loader = self.dev_loader

        test_golden = []
        test_preds = []
        f1s = []
        medical_jargons = set()

        # model = temp_model.model
        # model.to(device)
        # torch.cuda.current_device()
        # dev_loader = temp_model.dev_loader

        TF_flag = self.BINARY_model.TF_flag
        MLM_flag = self.BINARY_model.MLM_flag
        Binary_flag = self.BINARY_model.Binary_flag

        if normalization:
            normalizer = Normalizer(test_split, normalization_type = normalization_type)

            test_split = data_normalization(test_split, normalizer, target_entities = 'UMLS_concepts')
            test_split = data_normalization(test_split, normalizer, target_entities = 'entities')
        test_dataset = JargonTerm(test_split, tokenizer, BIOES_labeler, MAX_TOKEN_LEN, Binary_flag, TF_flag, MLM_flag, UMLS_labeler)
        test_loader = DataLoader(test_dataset, batch_size = bz)


        for batch in test_loader:
            torch.cuda.empty_cache()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            additional_features = {}
            if Binary_flag or TF_flag or MLM_flag:
                # UMLS_flags = batch['UMLS_flags']
                weighted_features = torch.tensor([])
                bin_weights = batch['bin_weights']
                additional_features['binary_features'] = bin_weights.to(device)

                if TF_flag:
                    TF_weights = batch['TF_weights']
                    weighted_features = torch.cat((weighted_features, TF_weights), dim = -1)
                if MLM_flag:
                    MLM_weights = batch['MLM_weights']
                    weighted_features = torch.cat((weighted_features, MLM_weights), dim = -1)
                additional_features['weighted_features'] = weighted_features.to(device)
                #print(additional_features)
            # print(additional_features.shape)
            # print(type(additional_features))
            sentids = batch['sentid']

            
            #optimizer.zero_grad()
            outputs = model(input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels, 
                            token_type_ids = token_type_ids, 
                            additional_features = additional_features)

            loss = outputs[0]
            preds = outputs[1]#.cpu().detach().numpy()
            # dev_losses.append(loss.cpu().data)
            #print(preds)

            # probs = softmax(logits).cpu().data
            # preds = torch.argmax(probs, dim=-1).cpu().detach().numpy()
            numpy_input_ids = input_ids.cpu().data.numpy()
            for batch_index, pred in enumerate(preds):
                labels = [BIOES_labeler.id2label[p] for p in pred]
                #print(input_ids[batch_index].cpu())
                input_ids = [tokenizer.convert_ids_to_tokens([input_id])[0] for input_id in numpy_input_ids[batch_index][:len(labels)]]

                # print(input_ids)
                # print(labels)
                # print(len(labels))
                jargons = BIOES_decode(input_ids, labels, tokenizer)

                for jargon in jargons:
                    #print(jargon)
                    medical_jargons.add(jargon['text'])

        #del(model)

        return medical_jargons
    


def sentencify(text):
    mednlp = medspacy.load()
    sents = [sent.text.strip() for sent in mednlp(text).sents if sent.text]
    return sents

def load_sents(sents):
    
    fout = open(os.path.join(DEFAULT_PATH,'temp/temp.tsv'),'w')
    
    for sent in sents:
        words = word_tokenize(sent)
        
        for word in words:
            fout.write("%s\t%s\n"%(word, 'O'))
        fout.write('\n')
        
    fout.close()
    
    note_aid_data_dict = load_ner_file(os.path.join(DEFAULT_PATH, 'temp/temp.tsv'))
    
    return note_aid_data_dict
    

def BIOES_decode(tokens, labels, tokenizer = None):
    # Output [{entity_type="", text = "", entity_token_span=(start,end)}]
    assert len(tokens) == len(labels), "the length of tokens and labels should be same."
    entity_list = []

    inner = False
    for i, (token, label) in enumerate(zip(tokens, labels)):
      label_type = label[0]
    
      #print(token, label)
      if len(label) > 0: 
        entity_type = label[2:]
      else:
        entity_type = ""
      
      # Type 1, not inner 
      #print(label_type)
      if not inner:
        if label_type == 'B':
          inner = (i, entity_type)
        elif label_type == 'S':
          entity_list.append((i, i+1, entity_type))
        else: continue;
      else:
        if label_type == 'B' or label_type == 'S':
          inner = False; continue;
        elif inner[1] != entity_type:
          inner = False; continue;
        elif label_type == 'E':
          start = inner[0]; end = i+1
          inner = False; entity_list.append((start, end, entity_type))
        else:
          continue
    #print(entity_list)
    entities = []
    for entity in entity_list:
      start = entity[0]; end = entity[1]; entity_type = entity[2]
      if not tokenizer:
        text = " ".join(tokens[start: end])
      else:
        text = tokenizer.convert_tokens_to_string(tokens[start: end])
      entities.append({ 'entity_type': entity_type,
                        'entity_token_span': (start, end),
                        'start_token': start,
                        'end_token': end,
                        'text': text})

    return entities


BINARY_model = evaluation_model(BINARY_PATH, data_splits, tokenizer, BIOES_labeler, UMLS_labeler)


# medjex = MedJEx(BINARY_model, device)

# medjex.predict('Fertilization begins when sperm binds to the corona radiata of the egg. Once the sperm enters the cytoplasm, a cortical reaction occurs which prevents other sperm from entering the oocyte. The oocyte then undergoes an important reaction. What is the next reaction that is necessary for fertilization to continue?')