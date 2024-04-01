
import pandas as pd
import numpy as np
import argparse
import pickle

from utils import *
config = load_config()

PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int)
    parser.add_argument("--type", type=str, help="zeroshot or fewshot")
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = arguments()
    topn = args.top
    prompttype = args.type

    with open(DATA_PATH.joinpath(f"top{topn}_{prompttype}_results.pkl"), 'rb') as f :
        prompt_results = pickle.load(f)
        prompt_results = list(map(lambda x : x[0]['generated_text'], prompt_results))



    # load original data too
    with open(DATA_PATH.joinpath("processed_ranking_datasets.pkl"), 'rb') as f :
        top3, top5, top10, dataset = pickle.load(f)


    if topn == 3 : 
        rankings = top3 
    elif topn == 5 : 
        rankings = top5
    else :
        rankings = top10

    # dataset['pred'] = prompt_results

    # rankings = rankings[rankings.color == 'y'].copy()
    # rankings = rankings.sort_values(by=['fileid', 'ranking']).drop_duplicates(subset=['fileid', 'phrase'])
    # rankings = rankings.sort_values(by=['fileid', 'ranking']).drop_duplicates(subset=['fileid', 'ranking'])
    # rankings = rankings[['fileid','phrase']].groupby('fileid', as_index=False).agg(lambda x : x )

    # rankings = rankings.rename(columns = {'fileid' : 'noteid'})

    # merged = pd.merge(dataset, rankings, on = 'noteid')

    precision, recall, mrr = [], [], []
    row_cnts = len(dataset)

    for idx in range(0,row_cnts)  :
        print("index is : ", idx)
        noteid = dataset.iloc[idx]['noteid']
        print("Gold : ", rankings[rankings.fileid == noteid][['phrase','ranking']].sort_values(by='ranking').reset_index(drop=True))
        print("\n\nPred : ", prompt_results[idx])

        val = input(f"\n\ninsert precision, recall, mrr : ")
        val = val.strip().split(' ')
        p, r, m = tuple(val)

        precision.append(float(eval(p)))
        recall.append(float(eval(r)))
        mrr.append(float(eval(m)))
        os.system('clear')
    
    precision = np.array(precision)
    recall = np.array(recall)
    mrr = np.array(mrr)

    save_this_results = (precision, recall, mrr)

    with open(DATA_PATH.joinpath(f"ranking_results_top{topn}_{prompttype}.pkl"), 'wb') as f :
        pickle.dump(save_this_results, f)

    print("the precision is ", precision.mean())
    print("the recall is ", recall.mean())
    print("the mrr is ", mrr.mean())