
import pickle
from pathlib import Path
import os, sys
import yaml

CURRENT_FILE_PATH = Path(__file__).absolute()
PROJECT_PATH = CURRENT_FILE_PATH.parents[1]
PROMPT_PATH = CURRENT_FILE_PATH.parents[1].joinpath('prompts')

class Config :

    def __init__(self, config_file) :
        self.file = config_file

    @property 
    def project_path(self) :
        return Path(self.file['project_path'])

    def model_path(self, model_name) :
        return self.file['model_path'][model_name]
    
    def template(self,inference, topn, **kwargs) :
        name = kwargs.get("prompt_name")

        if name == "" :
            template_path = PROMPT_PATH.joinpath(f'{inference}_{topn}.txt')
        else :
            template_path = PROMPT_PATH.joinpath(f'{name}_{inference}_{topn}.txt')

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
    