
import pickle
from pathlib import Path
import os, sys
import yaml

CURRENT_FILE_PATH = Path(__file__).absolute()
PROMPT_PATH = CURRENT_FILE_PATH.parents[1].joinpath('prompts')

class Config :

    def __init__(self, config_file) :
        self.file = config_file

    @property 
    def project_path(self) :
        return Path(self.file['project_path'])

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


