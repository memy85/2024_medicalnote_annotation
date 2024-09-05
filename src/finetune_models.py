#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import io
import json
import torch
import transformers
from transformers import EarlyStoppingCallback, IntervalStrategy
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig 
import typing
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
import ipdb
import random

from utils import *
# ipdb.set_trace()

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
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

# LOGGING_PATH = PROJECT_PATH.joinpath("logs/mistral)
# logging.basicConfig()


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    use_flash: bool = field(
        default=False,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_gpt_neo: bool = field(
        default=False,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help" : "Path to the evaluation data."})
    num_samples_used: int = field(
        default=None,
        metadata={"help": "Number of samples used in the instructions data, default is all"},
    )
    category: str = field(default="all", metadata={"help": "category of the instruction examples used"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=20000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    do_lora: bool = False
    load_in_8bit: bool = False
    load_best_model_at_end: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = None
    lora_weight_path: str = ""


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel
    ):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, num_samples_used=data_args.num_samples_used, category=data_args.category)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.eval_data_path :
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path, num_samples_used=data_args.num_samples_used, category=data_args.category)
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    else :
        return dict(train_dataset=train_dataset, data_collator=data_collator)


def train(args = None):
    if not args :
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    else :
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses(args.split())

    project_config = load_config()
    model_path = project_config.model_path(model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        # model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=training_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.use_flash = model_args.use_flash
    config.use_gpt_neo = model_args.use_gpt_neo

    if model_args.tokenizer_name_or_path is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            # model_args.model_name_or_path,
            model_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        # model_args.model_name_or_path,
        model_path,
        cache_dir=training_args.cache_dir,
        config=config,
        device_map="auto",
        load_in_8bit=training_args.load_in_8bit,
    )

    # if "llama" in model_args.model_name_or_path.lower() or "alpaca" in model_args.model_name_or_path.lower() or "vicuna" in model_args.model_name_or_path.lower():
    #     tokenizer.add_special_tokens({
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #     })
    #     tokenizer.pad_token = tokenizer.eos_token

    # if "mistral" in  model_args.model_name_or_path.lower() or "pythia" in model_args.model_name_or_path.lower() or "rwkv" in model_args.model_name_or_path.lower() or "opt" in model_args.model_name_or_path.lower():

    if "mistral" in  model_path.lower() or "pythia" in model_path.lower() or "rwkv" in model_path.lower() or "opt" in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token

    if training_args.do_lora:
        if lora_args.lora_weight_path == "":
            # load from scratch
            if training_args.gradient_checkpointing:
                model.enable_input_require_grads()

            if isinstance(lora_args.lora_target_modules, str):
                target_modules = lora_args.lora_target_modules.split(",")

            else:
                target_modules = None

            print("target modules: ", target_modules)

            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        else:
            # load from checkpoint
            model.enable_input_require_grads()
            peft_model_id = lora_args.lora_weight_path
            model = PeftModel.from_pretrained(model, peft_model_id)

            if training_args.gradient_checkpointing:
                model.get_input_embeddings().requires_grad_(True)

            # model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print("starts training ...")
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    return trainer


if __name__ == "__main__":
    train()
