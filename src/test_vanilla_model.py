import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model
from datasets import Dataset
import evaluate_model as em
import argparse
from openai import OpenAI
# import anthropic
import random
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("models/")

def load_openai(model_name, prompt) :
    api_key = os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
            model=model_name,
            input=[{'role' : 'user', 'content' : prompt}]
            )

    return

# def load_claude(model_name : str, prompt : str) :
#     client = anthropic.Anthropic()
#     response = client.messages.create(
#             model=model_name,
#             max_tokens = 200,
#             messages = [
#                 {'role' : 'user', 'content' : prompt}
#                 ]
#             )
#     return response.content[0].text

def process_texts(samples, template, **kwargs) :

    examples_dict = kwargs["inputs"]
    texts = samples['text']
    formated_texts = []
    for text in texts :
        examples_dict["note"] = text
        new_text = template.format(**examples_dict)
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


def run_inference(filtered_notes, 
                top10_dataset,
                  shot, 
                  prompt_name,
                  train_idx,
                  test_idx,
                  model_name,
                  model,
                  tokenizer,
                  cv_idx,
                  config
                  ) :

    if shot == "fewshot" :
        example_ids = filtered_notes.iloc[train_idx].sample(2, random_state=42)["noteid"].tolist()

        # prepare examples
        examples_dict = prepare_examples(example_ids, filtered_notes, top10_dataset)
    else :
        examples_dict = {}

    # test_file_ids = cv10[cv_idx]
    test_file_ids = filtered_notes.iloc[test_idx]['noteid'].tolist()

    # ============================ LOAD DATASETS =============================
    # shot = 'zeroshot'
    # prompt_name="generic"
    template = config.template(prompt_type=prompt_name, shot=shot)
    testset = Dataset.from_pandas(filtered_notes.iloc[test_idx])
    # testset = dataset.filter(lambda x : x["noteid"] in test_file_ids)

    testset = testset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": template, "inputs" : examples_dict})
    
    
    torch.cuda.empty_cache()
    # questions = testset['questions']
    questions = testset.to_pandas()['questions'].tolist()
    outputs = []
    
    inputs = tokenizer(questions, return_tensors='pt', padding=True)
    # inputs = tokenizer(questions, return_tensors='pt', padding=True)
    checker = list(map(lambda x : tokenizer.model_max_length < len(x), inputs))
    print("model max length : ", tokenizer.model_max_length)
    if any(checker) :
        raise ValueError("Input length exceeds model maximum length.")
    inputs.to("cuda")
    # model.to(devices)

    # =========================== START INFERENCE ============================
    max_new_token=200
    outputs = model.generate(**inputs, 
                max_new_tokens = max_new_token,
                temperature=0.1)
    
    arr_output = outputs.detach().cpu().numpy()
    start_of_generate_index = inputs.input_ids.shape[1]

    pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
    
    output_texts = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    
    with open(DATA_PATH.joinpath(f"{model_name}_{cv_idx}_{shot}_{prompt_name}_prediction.pkl"), 'wb') as f :
        pickle.dump(output_texts, f)


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    
    args = parser.parse_args()
    return args

def main() : 
    args = parse_arguments()
    model_name = args.model

    print("torch visible devices : ", torch.cuda.device_count(), file=sys.stderr)

    # ** whether to do finetuning from scratch 
    real_name = f"{model_name}"

    # ============================ LOAD DATASETS =============================

    # cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    cv10, top10_dataset, filtered_notes = load_cv_dataset()

    # cv_idx=0
    # =========================== LOAD MODEL ============================
    # model_name='mistral7b'
    # cv_idx=0
    # finetune = True
    
    devices = [i for i in range(torch.cuda.device_count())]

    base_model_path = config.model_path(model_name)

    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    for shot in ['zeroshot', 'fewshot'] :
        for prompt_name in ['generic', 'structured'] :
            for cv_idx, (train_idx, test_idx) in enumerate(cv10) :
                
                run_inference(filtered_notes, top10_dataset, shot, prompt_name, train_idx, test_idx, model_name, model, tokenizer, cv_idx, config)

if __name__ == "__main__"  :
    main()
