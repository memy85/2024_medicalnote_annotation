
# Load the tokenizer
#%%
import os
import pickle
import copy
import gc
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (pipeline, 
                          AutoModelForCausalLM, 
                          AutoModel,
                          AutoModelForSeq2SeqLM,
                          BioGptForCausalLM,
                          BioGptTokenizer, 
                          AutoTokenizer, 
                          Trainer, 
                          TrainingArguments,
                          DataCollatorForLanguageModeling) 

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from peft.peft_model import PeftModel
from peft.config import PeftConfig
from peft import LoraConfig, get_peft_model

from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
import sys
import argparse
from tqdm import tqdm
import json
from utils import *

import finetune_models
import evaluate_model as em


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts")

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
IGNORE_INDEX=-100
max_length=1024

#%%
def process_texts(samples, template) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = template.format(
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}
#%%

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _tokenize_fn(strings, tokenizer) :
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def load_biogpt() :
    MODEL_PATH = config.model_path("biogpt")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu")
    print("loaded model", file=sys.stderr)

    return tokenizer, model


def preprocess(
    sources ,
    targets ,
    tokenizer) : 
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model")
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = parse_arguments()
    model_name = args.model

    # ** whether to do finetuning from scratch 

    # print("The MODEL_PATH is ", MODEL_PATH, file=sys.stderr)

    #%%
    # ** explanation
    # cv5 : the cross validation training set list 
    # top10 dataset : medical key word dataset
    # filtered_notes : the EHR notes 
    cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))

    #%%

    # ================================================================================================================ #
    # =============================================== starting CV ==================================================== #
    # ================================================================================================================ #

    dataset = Dataset.from_pandas(filtered_notes)

    results = []
    # ** explanation 
    # cv5 : a list of cross-validation testset files, 83 files are testset
    # * original data : 104 = 83 (test) + 21 (train)
    # trainset : each files that belongs to the testset
    #%%
    for cv_idx, testset in enumerate(cv10) : 

        print(f"starting cv ------------------ {cv_idx} ----------------", file=sys.stderr)
        # filter_dataset : the test set
        dataset_name = f"cv{cv_idx}"
        trainset = dataset.filter(lambda x : x["noteid"] not in testset)

        with open(DATA_PATH.joinpath(f"finetune/{dataset_name}.json"),'r') as f :
            trainset = json.load(f)

        trainset = Dataset.from_list(trainset)

        # ** processed_dataset(testset) is the test set (83 files)
        # ** do finetuning if we have finetune flag
        # these are the train set (21 files)

        # ** Name of the finetuned model
        print("doing finetuning & not just evaluation", file=sys.stderr)
        finetune_model_name = f"{model_name}_finetuned_{dataset_name}"
        if "biogpt" in model_name :
            tokenizer, model = load_biogpt()
            model.cuda()
        else :
            MODEL_PATH = config.model_path(model_name=model_name)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer.pad_token=tokenizer.eos_token


        # ===================================================
        # ======================================== Load Peft
        # ===================================================

        def tokenize_function(examples):
            model_inputs = [tokenizer.tokenize(input) for input in examples['input']]
            outputs = [tokenizer.tokenize(output) for output in examples["output"]]
            merged_texts = []
            for input, output in zip(model_inputs, outputs) :
                max_input_length = max_length - len(outputs) -1
                truncated_input_tokens = input[:max_input_length]

                reformated_input_token = truncated_input_tokens + ["\n"] + output
                # print("reformated input", reformated_input_token, file=sys.stderr)
                text = tokenizer.convert_tokens_to_string(reformated_input_token)
                merged_texts.append(text)

            examples['merged_texts'] = merged_texts
            # print("merged texts", merged_texts, file=sys.stderr)
            model_inputs = tokenizer(examples['merged_texts'], padding="max_length", truncation=True, max_length=max_length)
            return model_inputs

        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Set to False for causal language modeling
            )

        # Tokenizing input and output text
        tokenized_datasets = trainset.map(tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"./models/{model_name}-finetuned-{cv_idx}",
            # do_train=True,
            learning_rate=3e-5,
            per_device_train_batch_size=4,
            # evaluation_strategy="epoch",
            # per_device_eval_batch_size=8,
            num_train_epochs=40, # 100
            save_steps=10,
            save_total_limit=1,
            report_to='tensorboard',
            weight_decay=0.01,
            logging_dir=f'./logs/base_model_train_logs/{model_name}',
            logging_steps=1,
            # load_best_model_at_end=True,
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator, # data collator helps creates labels for causal language modeling 
        )

        # Train the model
        trainer.train()
        # model = trainer.model
        # tokenizer = trainer.tokenizer
        trainer.save_state()
        trainer.save_model(output_dir=f"./models/{model_name}-finetuned-{cv_idx}")

        del model
        del trainer
        torch.cuda.empty_cache()

