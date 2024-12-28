
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
    
    @property
    def file_path(self) :
        return PROJECT_PATH


    def model_path(self, model_name) :
        return self.file['model_path'][model_name]
    
    def template(self, topn, inference) :
        template_path = PROMPT_PATH.joinpath(f'{inference}_{topn}.txt')
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


def load_config() :
    project_path = CURRENT_FILE_PATH.parents[1]
    config_path = project_path.joinpath("config/config.yaml")

    with open(config_path) as f : 
        config = yaml.load(f, yaml.SafeLoader)

    return Config(config)


def format_prompt(fileids, topN, prompt) : 
    '''
    format prompts for the examples that are used for finetuning
    '''

    with open(PROJECT_PATH.joinpath("data/processed/cv_processed_ranking_datasets_new.pkl"), 'rb') as f :
        _, top10_dataset, notes = pickle.load(f)
    # get the EHR note

    if prompt == "modified" :
        with open(PROMPT_PATH.joinpath(f"{prompt}_finetune_instruction_top{topN}.txt"), 'r') as f :
            instruction = f.read()
    else :
        with open(PROMPT_PATH.joinpath(f"finetune_instruction_top{topN}.txt"), 'r') as f :
            instruction = f.read()

    # instruction = instruction.format(topN=topN)

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
    
