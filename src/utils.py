
from pandas import DataFrame
from typing import List, Tuple, Dict
import pandas as pd
import pickle
from pathlib import Path
import os, sys
import requests
import argparse
import yaml

CURRENT_FILE_PATH = Path(__file__).absolute()
PROJECT_PATH = CURRENT_FILE_PATH.parents[1]
PROMPT_PATH = CURRENT_FILE_PATH.parents[1].joinpath('prompts')
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

class Config :

    def __init__(self, config_file) :
        self.file = config_file

    @property 
    def project_path(self) :
        return Path(self.file['project_path'])

    def model_path(self, model_name) :
        return self.file['model_path'][model_name]
    
    def template(self, prompt_type, shot, **kwargs) :

        # name = kwargs.get("prompt_name")
        template_path = PROMPT_PATH.joinpath(f'{prompt_type}_{shot}_top10.txt')
        # template_path = PROMPT_PATH.joinpath(f'extraction_{shot}.txt')

        with open(template_path, 'r') as f :
            template = f.read()
        return template
    
    def get_data_path(self, dataset_name) :
        '''
        dataset names : mimic_discharge, annotated_examples_path
        '''
        if dataset_name == "mimic_discharge" :
            return Path(self.file['project_path']).joinpath("data/raw/discharge.csv")
        else :
            print("there are ...")
            print(os.listdir(self.file['annotated_examples']['directory']\
                              + self.file['annotated_examples']['notes']))
            return Path(self.file['annotated_examples']['directory']\
                        + self.file['annotated_examples']['notes']),\
                        Path(self.file['annotated_examples']['directory']\
                              + self.file['annotated_examples']['annotations'])

    @property 
    def device(self) :
        return self.file['device']

    @property 
    def checkpoint(self) :
        return self.file['checkpoint']

    @property 
    def restructuring_format(self) :
        path = PROMPT_PATH.joinpath("restructuring_instruction.txt")
        with open(path, 'r') as f :
            prompt = f.read()
        return prompt
    
    @property
    def chatgpt_prompt(self) :
        path = PROMPT_PATH.joinpath("chatgpt_prompt.txt")
        with open(path, 'r') as f :
            prompt = f.read()
        return prompt


def load_config() :
    project_path = CURRENT_FILE_PATH.parents[1]
    config_path = project_path.joinpath("config/config.yaml")

    with open(config_path) as f : 
        config = yaml.load(f, yaml.SafeLoader)

    return Config(config)


def format_prompt(fileids, topN) : 
    '''
    format prompts for the examples that are used for finetuning
    '''

    with open(PROJECT_PATH.joinpath("data/processed/cv_processed_ranking_datasets.pkl"), 'rb') as f :
        _, top10_dataset, notes = pickle.load(f)
    # get the EHR note

    with open(PROMPT_PATH.joinpath("finetune_instruction.txt"), 'r') as f :
        instruction = f.read()
    instruction = instruction.format(topN=topN)

    parsed_dataset = []
    for fileid in fileids : 
        ehr_note = notes[notes.noteid == fileid]
        input_text = ehr_note.values[0][2]

        # format the outputs 
        df = top10_dataset[(top10_dataset.fileid == fileid) & (top10_dataset.ranking < (topN+1))].copy()
        output_text = ""

        for idx, row in df.iterrows() :
            output_text += str(row.ranking) + " " + row.phrase + "\n" 

        output = {"input" : input_text,
                  "output" : output_text,
                  "instruction" : instruction
                  }

        parsed_dataset.append(output)

    return parsed_dataset

def send_line_message(message):
    # LINE Messaging API endpoint
    url = 'https://api.line.me/v2/bot/message/push'

    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
    }

    # Request body
    data = {
        'to': LINE_USER_ID,
        'messages': [
            {
                'type': 'text',
                'text': message
            }
        ]
    }
    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print(f'Failed to send message. Status code: {response.status_code}, Response: {response.text}')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_cv_dataset() -> Tuple[List, DataFrame, DataFrame] :

    DATA_PATH = PROJECT_PATH.joinpath("data/processed")

    final_cv_set = pd.read_pickle(DATA_PATH.joinpath("final_cv_set.pkl"))
    final_notes = pd.read_json(DATA_PATH.joinpath("final_notes.json"), lines=True, orient="records") 
    final_jargons = pd.read_json(DATA_PATH.joinpath("final_jargons.json"), lines=True, orient="records")  

    return final_cv_set, final_jargons, final_notes 


def prepare_examples(example_ids : List[str], notes : DataFrame, jargons : DataFrame) -> Dict :  

    example_dict = {}
    for idx, noteid in enumerate(example_ids) : 
        note = notes[notes.noteid == noteid]['text'].iloc[0]
        jargon_dataset = jargons[jargons.noteid == noteid].sort_values(by="ranking")

        outputs = ""
        for i, row in jargon_dataset.iterrows() :
            rank = row["ranking"]
            term = row["jargon"]
            outputs += f"{rank}. {term}\n"
        
        example_dict[f"example_note{idx+1}"] = note
        example_dict[f"example_output{idx+1}"] = outputs

    return example_dict



