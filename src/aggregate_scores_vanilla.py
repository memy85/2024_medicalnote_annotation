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

def load_results(cv_idx, real_name) :

    with open(DATA_PATH.joinpath(f"{real_name}_{cv_idx}_{shot}_{prompt_name}_prediction.pkl"), "rb") as f :
        results = pickle.load(f)

    return results 

def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    parser.add_argument("--finetune", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--mimic", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--shot", type=str, help='choose between zeroshot, fewshot', default='zeroshot')
    parser.add_argument("--topn", type=str, help='choose between default top3, top5, top10', default='top3')
    parser.add_argument("--baseline", type=str2bool, nargs='?', const=True, default=False, help="whether this is a baseline model or not")
    parser.add_argument("--prompt_name", type=str, help="generic or structured", default=None)
    
    args = parser.parse_args()
    return args


def main() :
    args = parse_arguments()
    model_name = args.model
    # cv_idx = args.cv_idx

    # real_name = f"{model_name}_finetuned"

    #%%
    # cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    cv10, top10_dataset, filtered_notes = load_cv_dataset()

    for rank in [3,5,10] :
        topn = f"top{rank}"

        results = []
        for cv_idx, (train_idx, test_idx) in enumerate(cv10) :

            # test_file_ids = cv10[cv_idx]
            test_file_ids = filtered_notes.iloc[test_idx]["noteid"].tolist()

            #%% LOAD RESULTS
            pred_dataset = load_results(cv_idx, real_name, shot, prompt_name)

            #%% 
            gold_dataset = top10_dataset[top10_dataset.noteid.isin(test_file_ids)].copy()
            gold_dataset = gold_dataset[gold_dataset.ranking < rank + 1].copy()
            testset_notes = filtered_notes[filtered_notes.noteid.isin(test_file_ids)].copy()

            #%% calculate results
            p_avg, r_avg, f1_avg, m_avg, map_avg = em.get_results(gold_dataset, pred_dataset, test_file_ids, rank) 

            #%%
            if do_finetune : 
                record = {"model" : real_name, "cv" : cv_idx, "topN" : rank, "shots" : shot, "precision" : p_avg, "recall" : r_avg, "f1" : f1_avg, "mrr" : m_avg, "map" : map_avg}
                results.append(record)

            else :
                record = {"model" : real_name, "cv" : cv_idx, "topN" : rank, "shots" : shot, "precision" : p_avg, "recall" : r_avg, "f1" : f1_avg, "mrr" : m_avg, "map" : map_avg}
                results.append(record)

        evaluation_df = em.calculate_the_scores_with_confidence_intervals(results, baseline_flag) 

        evaluation_df.to_pickle(DATA_PATH.joinpath(f"{topn}_{real_name}_{shot}_{prompt_name}_df.pkl"))

        with open(DATA_PATH.joinpath(f"{topn}_{real_name}_{shot}_{prompt_name}_evaluation.pkl"), 'wb') as f :
            pickle.dump(results, f)
        
    print(f"finished!")



if __name__ == "__main__"  :
    main()
