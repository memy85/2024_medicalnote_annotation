import re
import numpy as np
import pandas as pd
import argparse
import torch
from utils import *
import random
from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel
import evaluate_model as em


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
OUTPUT_PATH = PROJECT_PATH.joinpath("outputs")

# set tokens
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# get example 
cv5, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets.pkl"))

# need to filter out only test dataset that were not in the train set
# 1. let's filter out the test dataset
wholedataset = set(top10_dataset.fileid.unique().tolist())
trainset = set(cv5[0])
testset = list(wholedataset - trainset)

# set random samples
random.seed(1)
sample_files = random.sample(testset, 3)
print("The chosen files are : ", file=sys.stderr)
print(sample_files, file=sys.stderr)

def process_texts(samples, template) :

    texts = samples['text']
    formated_texts = []
    for text in texts :
        new_text = template.format(
            context = text
        )
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


# process ranking
def process_ranking(sample_file) :
    ranking = top10_dataset[top10_dataset.fileid == sample_file]['ranking']
    return ranking

# process notes
def process_notes(sample_file) :
    notes = filtered_notes[filtered_notes.noteid == sample_file]['text']
    notes = notes.values.tolist()[0]
    return notes

# process gold label data
def process_gold_label(sample_file) :
    temp = top10_dataset[top10_dataset.fileid == sample_file][['phrase', 'ranking']].sort_values(by="ranking")
    gold_answer = ""
    for _, row in temp.iterrows() :
        gold_answer += str(row['ranking']) + ' ' + row['phrase'] + '\n'
    return gold_answer


def infer_answers(text, model, tokenizer, max_tokens) :
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    print("length of input ids are : ", len(input_ids), file=sys.stderr)
    output_ids = model.generate(input_ids=input_ids, 
                                max_new_tokens = max_tokens)
    
    arr_output = output_ids.detach().cpu().numpy()
    start_of_generate_index = input_ids.shape[1]
    pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
    return pred_output

def write_outputs(model_name, text):
    with open(OUTPUT_PATH.joinpath(f"{model_name}_reformat_output.txt"), 'w') as f :
        f.write(text)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model")
    parser.add_argument("--finetune", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--inference", type=str, help='choose between zeroshot, fewshot', default='zeroshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10', default='top3')
    parser.add_argument("--max_new_token", type=int, help='max token', default=100)
    parser.add_argument("--quantization", type=str, help="choose either 8 or None", default=None)
    
    args = parser.parse_args()
    return args

def main() :

    args = parse_arguments()
    MODEL_PATH = config.model_path(args.model)
    finetune_flag = args.finetune
    inference = args.inference
    topn = args.topn
    max_new_token = args.max_new_token
    rank = int(topn.replace("top", ""))
    
    # set file name if finetuned
    if finetune_flag :
        file_name = args.model + f"_finetuned_{topn}_{inference}"
    else :
        file_name = args.model + f"_{topn}_{inference}"


    # process dataset with questions
    dataset = Dataset.from_pandas(filtered_notes)
    template = config.template(topn=topn, inference=inference)
    restructuring_template = config.restructuring_format
    dataset = dataset.map(process_texts, batched=True, fn_kwargs ={"template" : template})
    dataset = dataset.to_pandas()

    # load model
    if finetune_flag :
        model_name = args.model + "_finetuned"
        MODEL_PATH = config.model_path(model_name)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
        model = PeftModel.from_pretrained(model, MODEL_PATH)
    else :
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    tokenizer.pad_token = tokenizer.eos_token

    output_texts = ""
    for idx, sample_file in enumerate(sample_files) :
        text = dataset[dataset.noteid == sample_file]['questions'].values.tolist()[0]
        answers = infer_answers(text=text, 
                                model=model, 
                                tokenizer=tokenizer, 
                                max_tokens=max_new_token)
        formated_answers = infer_answers(text=restructuring_template.format(output=answers),
                      model=model,
                      tokenizer=tokenizer,
                      max_tokens=max_new_token)
        
        print(f"answers : {answers}", file=sys.stderr)
        pred = em.preprocess_outputs_of_model(formated_answers, rank)

        # calculate precision/recall/f1-score/mrr
        p = re.compile("^\d+")
        gold = top10_dataset[top10_dataset.fileid == sample_file][['ranking', 'phrase']].copy()
        gold['ranking'] = gold['ranking'].apply(lambda x : int(p.findall(str(x))[0]))
        gold = [tuple(x) for x in gold.to_numpy()]

        precision, recall = em.calculate_precision_recall(gold, pred=pred)
        precision = round(precision, 3)
        recall = round(recall, 3)
        print(f"precision : {precision}\nrecall : {recall}", file=sys.stderr)
        try :
            f1_score = round(2*precision*recall/(precision+recall),3)
        except ZeroDivisionError :
            f1_score = 0
        mrr = em.calculate_mrr(gold, pred)


        #%% format outputs
        gold_text = ""
        for num, content in gold :
            if int(num) <= rank :
                gold_text += f"{num}. {content}\n"
                
        pred_text = ""
        for num, content in pred :
            pred_text += f"{num}. {content}\n"

        
        output_texts += f"\n{sample_file} ================================= \n\n\n" + \
                        f"precison : {precision}\nrecall : {recall}\nf1_score : {f1_score}\nmrr : {mrr}" + \
                        f"\n\ngold : \n {gold_text}\n \npred : \n {pred_text}"
        

    # write the answers with the score
    write_outputs(model_name=file_name, text=output_texts)

if __name__ == "__main__" :
    main()


    

        

