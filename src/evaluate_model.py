import sys
import pickle
import re
import pandas as pd
import numpy as np
import argparse

from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts")

def load_results(file_name) :
    '''
    file example : mistral_top3_zeroshot.pkl or mistral_top3_fewshot.pkl
    '''

    with open(DATA_PATH.joinpath(file_name), 'rb') as f :
        results = pickle.load(f)

    return results 


def load_datasets(topN) :
    '''
    returns topN and testset
    '''

    with open(DATA_PATH.joinpath("processed_ranking_datasets.pkl"), 'rb') as f :
        top3, top5, top10, dataset, _, _ = pickle.load(f)

    if topN == 3 :
        return top3, dataset
    elif topN == 5 :
        return top5, dataset
    else :
        return top10, dataset
    

def preprocess_outputs_of_model(text, rank) :
    p = re.compile('(\d+)\.\d?\.?\s(.+)')
    output = p.findall(text)

    numbers = list(map(lambda x : x[0], output))
    output = list(map(lambda x : x[1].strip().lower(), output))

    output = [(x,y) for x, y in zip(numbers, output)]
    output = list(filter(lambda x : int(x[0]) < (rank + 1), output))

    return output


def calculate_precision_recall(gold, pred) :
    '''
    using exact match, we calculate the precision
    '''
    gold_extracted = list(map(lambda x : x[1], gold))
    pred_extracted = list(map(lambda x : x[1], pred))

    cnt = 0
    for gold_element in gold_extracted : 
        for pred_element in pred_extracted :
            if gold_element.lower() in pred_element.lower() :
                cnt += 1

    pred_cnt = len(pred_extracted)
    gold_cnt = len(gold_extracted)

    if pred_cnt == 0 :
        pred_cnt = 0.001

    precision_score = cnt / pred_cnt

    if precision_score > 1.0 :
        precision_score = 1.0

    if gold_cnt == 0 :
        gold_cnt = 0.001

    recall_score = cnt / gold_cnt

    if recall_score > 1.0 :
        recall_score = 1.0

    return precision_score, recall_score



def calculate_mrr(gold, pred) :
    '''
    calculates the MRR
    '''
    gold_extracted = list(map(lambda x : x[1], gold))
    pred_extracted = list(map(lambda x : x[1], pred))

    for i, gold_element in enumerate(gold_extracted) : 
        for j, pred_element in enumerate(pred_extracted) :
            if gold_element.lower() in pred_element.lower() :
                # print("the gold rank is : ", gold[i][0])
                # print("the pred rank is : ", pred[j][0])
                if int(gold[i][0]) == int(pred[j][0]) :
                    # print("yes!")
                    print("the rank is : ",gold[i][0],file=sys.stderr)
                    return 1 / int(gold[i][0])
                else :
                    continue
            else :
               continue 
    # print("no..")
    return 0


def get_results(gold_dataset, pred_dataset, testset_ids, rank) :
    '''
    gold_dataset : top3, top5, top10 key medical terms annotated by medical specialists
    pred_dataset : outputs of the models. The sequence is the same as the testset_ids
    testset_ids : the test dataset ids list
    '''
    fileids = testset_ids
    p = re.compile('^\d+')

    precisions = []
    recalls = []
    mrrs = []
    for idx, fileid in enumerate(fileids) : 
        # process gold labels
        gold = gold_dataset[gold_dataset.fileid == fileid][['ranking', 'phrase']].copy()
        gold['ranking'] = gold['ranking'].apply(lambda x : int(p.findall(str(x))[0]))
        gold = [tuple(x) for x in gold.to_numpy()]

        # process prediction dataset
        pred = pred_dataset[idx]
        pred = preprocess_outputs_of_model(pred, rank)

        precision, recall = calculate_precision_recall(gold, pred)
        mrr = calculate_mrr(gold, pred)

        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
    
    p_avg = round(np.array(precisions).mean(),3)
    r_avg = round(np.array(recalls).mean(), 3)
    f1_avg = round(2*p_avg*r_avg/(p_avg+r_avg),3)
    m_avg = round(np.array(mrrs).mean(),3)

    print("The F1 score is %.3f" %f1_avg)
    print("The precision is %.3f" %p_avg)
    print("The recall is %.3f" %r_avg)
    print("The mrr is %.3f" %m_avg)

    return p_avg, r_avg, f1_avg, m_avg # precisions, recalls, mrrs

def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose the appropriate name of the model")
    args = parser.parse_args()
    return args

def main() :
    args = parse_arguments()
    model = args.model

    ranks = [3,5,10]
    shots = ['zeroshot', 'fewshot']

    scores = []
    for shot in shots :
        for rank in ranks : 

            file_name = model + f'_top{rank}' + f'_{shot}' + '.pkl'
            results = load_results(file_name=file_name)
            topn_dataset, dataset = load_datasets(rank)

            p, r, f1, m = get_results(topn_dataset, results, dataset, rank)

            scores.append({"model" : model, "topN" : rank, "shots" : shot, "score" : 'precision', "value" : p})
            scores.append({"model" : model, "topN" : rank, "shots" : shot, "score" : 'recall', "value" : r})
            scores.append({"model" : model, "topN" : rank, "shots" : shot, "score" : 'f1', "value" : f1})
            scores.append({"model" : model, "topN" : rank, "shots" : shot, "score" : 'mrr', "value" : m})
    
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(DATA_PATH.joinpath(f"{model}_evaluation_results.csv"), index=False)


if __name__ == "__main__" :
    main()


