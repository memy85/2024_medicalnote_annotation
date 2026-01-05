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
            note = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    parser.add_argument("--aug_size", type=int, help="augmentation size", default=None)
    
    args = parser.parse_args()
    return args

def main() : 
    args = parse_arguments()
    model_name = args.model
    aug_size = args.aug_size
    print("torch visible devices : ", torch.cuda.device_count(), file=sys.stderr)

    # cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    cv10, top10_dataset, filtered_notes = load_cv_dataset()

    # =========================== LOAD MODEL ============================
    # model_name='llama31'
    # cv_idx=0
    # finetune = True

    base_model_path = config.model_path(model_name)

    lora_path = MODEL_PATH.joinpath(f'{model_name}-mimic-finetuned-{aug_size}').as_posix()

    model = AutoModelForCausalLM.from_pretrained(lora_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    # model = PeftModel.from_pretrained(model,lora_path)
    # model.to('cuda')

    # ============================ LOAD DATASETS =============================
    # shot = 'zeroshot'
    # prompt_name=None
    if "llama" in model_name : 
        template = config.template(shot="zeroshot", prompt_type="structured")
    else : 
        template = config.template(shot="zeroshot", prompt_type="generic")

    dataset = Dataset.from_pandas(filtered_notes)

    # =========================== START INFERENCE ============================
    for cv_idx, (train_idx, test_idx) in enumerate(cv10) :
        # test_file_ids = cv10[cv_idx]
        test_file_ids = filtered_notes.iloc[test_idx]['noteid'].tolist()
        testset = dataset.filter(lambda x : x["noteid"] in test_file_ids)

        # ** processed_dataset(testset) is the test set (83 files)
        testset = testset.map(process_texts, batched=True, \
                                        fn_kwargs={"template": template})

        torch.cuda.empty_cache()
        # questions = testset['questions']
        questions = testset.to_pandas()['questions'].tolist()
        outputs = []
        
        inputs = tokenizer(questions, return_tensors='pt', padding=True, padding_side='left')
        inputs.to('cuda')
        max_new_token=200

        outputs = model.generate(**inputs, 
                    max_new_tokens = max_new_token,
                    temperature=0.1)
        
        arr_output = outputs.detach().cpu().numpy()
        start_of_generate_index = inputs.input_ids.shape[1]

        # pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
        
        output_texts = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)
        torch.cuda.empty_cache()

        with open(DATA_PATH.joinpath(f"{model_name}_mimic_{aug_size}_finetuned_{cv_idx}_zeroshot_prediction.pkl"), 'wb') as f :
            pickle.dump(output_texts, f)


if __name__ == "__main__"  :
    main()
