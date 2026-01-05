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
    parser.add_argument("--finetune", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--shot", type=str, help='choose between zeroshot, fewshot', default='zeroshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10', default='top10')
    parser.add_argument("--baseline", type=str2bool, nargs='?', const=True, default=False, help="whether this is a baseline model or not")
    parser.add_argument("--prompt_name", type=str, help="name of prompt", default=None)
    parser.add_argument("--cv_idx", type=int, help="index of CV", default=None)
    
    args = parser.parse_args()
    return args

def main() : 
    args = parse_arguments()
    model_name = args.model
    shot = args.shot
    do_finetune = args.finetune
    prompt_name = args.prompt_name
    baseline_flag = args.baseline
    cv_idx = args.cv_idx
    # rank = int(topn.replace("top", ""))
    print("finetune flag : ", do_finetune, file=sys.stderr)
    print("torch visible devices : ", torch.cuda.device_count(), file=sys.stderr)

    # ** whether to do finetuning from scratch 
    if model_name in ["llama31"] : 
        prompt_name = "generic"
    else : 
        prompt_name = "structured"

    if do_finetune : 
        real_name = f"{model_name}_finetuned"
        prompt_name = "generic"
        shot='zeroshot'

    else :
        real_name = f"{model_name}"


    # cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    cv10, top10_dataset, filtered_notes = load_cv_dataset()
    test_idx = cv10[cv_idx][1]
    test_file_ids = filtered_notes.iloc[test_idx]['noteid'].tolist()

    # ============================ LOAD DATASETS =============================
    # shot = 'zeroshot'
    # prompt_name="generic"
    if "llama" in model_name : 
        template = config.template(shot=shot, prompt_type="structured")
    else : 
        template = config.template(shot=shot, prompt_type="generic")
    dataset = Dataset.from_pandas(filtered_notes)
    testset = dataset.filter(lambda x : x["noteid"] in test_file_ids)

    # ** processed_dataset(testset) is the test set (83 files)
    testset = testset.map(process_texts, batched=True, \
                                    fn_kwargs={"template": template})

    # =========================== LOAD MODEL ============================
    # model_name='mistral7b'
    # cv_idx=0
    # finetune = True

    if do_finetune : 
        lora_path = MODEL_PATH.joinpath(f'{model_name}-finetuned-{cv_idx}').as_posix()
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(lora_path)
        # tokenizer.pad_token_id = tokenizer.eos_token_id
    else : 
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        # tokenizer.pad_token_id = tokenizer.eos_token_id

    
    # =========================== START INFERENCE ============================

    torch.cuda.empty_cache()
    # questions = testset['questions']
    questions = testset.to_pandas()['questions'].tolist()
    outputs = []
    
    inputs = tokenizer(questions, return_tensors='pt', padding=True, padding_side='left')
    inputs.to('cuda')
    model.to('cuda')
    max_new_token=400

    outputs = model.generate(**inputs, 
                max_new_tokens = max_new_token,
                temperature=0.1)
    
    arr_output = outputs.detach().cpu().numpy()
    start_of_generate_index = inputs.input_ids.shape[1]

    # pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
    
    output_texts = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    
    if do_finetune :
        with open(DATA_PATH.joinpath(f"{model_name}_finetuned_{cv_idx}_zeroshot_generic_prediction.pkl"), 'wb') as f :
            pickle.dump(output_texts, f)
    else : 
        with open(DATA_PATH.joinpath(f"{model_name}_{cv_idx}_{shot}_{prompt_name}_prediction.pkl"), 'wb') as f :
            pickle.dump(output_texts, f)



if __name__ == "__main__"  :
    main()
