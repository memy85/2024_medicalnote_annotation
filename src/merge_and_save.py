# MERGE AND SAVE THE MODEL
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, PeftModelForCausalLM
import argparse

from utils import * 

DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("models/")

config = load_config()
    
def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    args = parser.parse_args()
    return args

def load_base_model_path(base_model) : 
    base_model_path = config.model_path(base_model)
    return base_model_path

def load_model_path(model_name : str, cv_idx : int = None) : 
    
    if cv_idx is not None : 
        model_name = model_name + f"-{cv_idx}"

    model_path = MODEL_PATH.joinpath(model_name)
    return  model_path


def load_model_in_huggingface(model_name : str, cv_idx = None) :
    model_name = "llama31-finetuned"
    cv_idx = 0
    base_model_name = model_name.split("-")[0]
    base_model_path = load_base_model_path(base_model_name)

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # add lora
    model_path = load_model_path(model_name, cv_idx)
    lora_model = PeftModel.from_pretrained(base_model, model_path)

    return model, tokenizer


def save_model() : 


    return 


def main() :


    pass


if __name__ == "__main__" :
    main()
