# finetuning codes

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers import Trainer, AutoConfig 

import torch
from torch.utils.data import Dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
import argparse
from utils import *


config = load_config()
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
SAVE_PATH = PROJECT_PATH.joinpath("model/")

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
MAX_TOKEN_LENGTH = 8192

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
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


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 num_samples_used=None, 
                 category="all"):

        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        random.seed(10)

        if category != "all":
            cate_used = category.split(",")
            cate_data_dict = []
            for item in list_data_dict:
                if item['high_category'] in cate_used:
                    cate_data_dict.append(item)
            list_data_dict = cate_data_dict

        if num_samples_used is not None:
            list_data_dict = random.sample(list_data_dict, num_samples_used)

        print("number of instances used: ", len(list_data_dict), 'with category: ', category)
        random.shuffle(list_data_dict)
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path,
                                num_samples_used, category) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, num_samples_used=num_samples_used, category=category)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.eval_data_path :
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path, num_samples_used=data_args.num_samples_used, category=data_args.category)
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    else :
        return dict(train_dataset=train_dataset, data_collator=data_collator)

def load_dataset_path(cv_idx : int) :
    
    return 


def load_dataset() :
    
    # load the 10 fold datasets
    cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    # cv10 consists the list of notes that are the test sets

    return cv10, top10_dataset, filtered_notes

def curate_model(model) :
    '''
    prepare model for lora training
    '''

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="q_proj,v_proj,k_proj,o_proj".split(","),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    return model

    

def finetune_model(model, tokenizer, data_module, lora_weight_path=None) :

    if lora_weight_path : 
        model.enable_input_require_grads()
        peft_model_id = lora_weight_path

        model = PeftModel.from_pretrained(model, peft_model_id)
        model.get_input_embeddings().requires_grad_(True)

    else : 
        model.enable_input_require_grads() 
        model = curate_model(model)
        
    # prepare dataset
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print("starts training ...")
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=SAVE_PATH)
    pass

def arguments() :
    parser = argparse.ArgumentParser()
    parser.add_arguments("--model", type=str)
    parser.add_arguments("--cv", type=int)
    args = parser.parse_args()

    return args

def main() : 
    args = arguments()
    model_name = args.model
    cv_idx = args.cv
    
    # override save path 
    SAVE_PATH = SAVE_PATH.joinpath(f"{model_name}/{cv_idx}")

    model_path = config.model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              model_max_length=MAX_TOKEN_LENGTH,
                                              padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # curate dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # finetune model
    finetune_model(model, )


    pass


if __name__ == "__main__" :
    main()
