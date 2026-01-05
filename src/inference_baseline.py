import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model
from datasets import Dataset
import evaluate_model as em
import argparse
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("models/")

def process_texts(samples, template) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = template.format(
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    parser.add_argument("--cv_idx", type=int, help="index of CV", default=None)
    
    args = parser.parse_args()
    return args

def main() : 
    args = parse_arguments()
    model_name = args.model
    cv_idx = args.cv_idx
    print("torch visible devices : ", torch.cuda.device_count(), file=sys.stderr)

    # ** whether to do finetuning from scratch 
    real_name = f"{model_name}_finetuned"

    cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    # cv_idx=0
    test_file_ids = cv10[cv_idx]

    # ============================ LOAD DATASETS =============================
    # shot = 'zeroshot'
    # prompt_name="generic"
    template = config.template(shot=shot, prompt_name=prompt_name)
    dataset = Dataset.from_pandas(filtered_notes)
    testset = dataset.filter(lambda x : x["noteid"] in test_file_ids)

    # ** processed_dataset(testset) is the test set (83 files)
    testset = testset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": template})

    # =========================== LOAD MODEL ============================
    
    lora_path = MODEL_PATH.joinpath(f'{model_name}-finetuned-{cv_idx}').as_posix()
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    model = AutoModelForCausalLM.from_pretrained(lora_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    
    # =========================== START INFERENCE ============================

    torch.cuda.empty_cache()
    # questions = testset['questions']
    texts = testset.to_pandas()['text'].tolist()
    outputs = []
    
    max_length = 1024
    max_new_token = 200

    inputs = tokenizer(texts, return_tensors='pt', padding='max_length', padding_side='left', max_length-max_new_token)
    inputs.to('cuda')
    model.to('cuda')

    outputs = model.generate(**inputs, 
                max_new_tokens = max_new_token,
                temperature=0.1)
    
    arr_output = outputs.detach().cpu().numpy()
    start_of_generate_index = inputs.input_ids.shape[1]

    pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
    
    output_texts = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    
    with open(DATA_PATH.joinpath(f"{model_name}_{cv_idx}_prediction.pkl"), 'wb') as f :
        pickle.dump(output_texts, f)



if __name__ == "__main__"  :
    main()
