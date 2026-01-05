import pandas as pd
import torch
import torch.nn as nn
from typing import List
from peft import PeftModel, get_peft_model
from datasets import Dataset
import evaluate_model as em
import argparse
from openai import OpenAI, AsyncOpenAI, RateLimitError
import anthropic
import asyncio
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("models/")

class RateLimitError(Exception): pass

async def load_client(provider) : 
    if 'openai' == provider: 
        api_key = os.environ['OPENAI_API_KEY']
        org_id = os.environ['OPENAI_ORG_ID']
        client = AsyncOpenAI(api_key=api_key, organization=org_id)
        
    elif 'anthropic' == provider : 
        client = anthropic.AsyncAnthropic()

    return client


async def call_one(client, provider, prompt, semaphore, model) :
    async with semaphore :
        try :
            if provider == "openai" :
                response = await client.chat.completions.create(model=model,
                                                                messages=[{'role' : 'user' , 'content' : prompt}],
                                                                max_completion_tokens=200, 
                                                                temperature=0.1)
                return response.choices[0].message.content

            else :
                response = await client.messages.create(model=model, max_completion_tokens=200, messages=[{'role' : 'user' , 'content' : prompt}], temperature=0.1)
                return "".join(block.text for block in response.content if getattr(block, "type", None))

        except Exception as e : 
            raise RateLimitError(e)

async def run_batch(prompts : List[str], provider, model) :
    client = await load_client(provider)
    sem = asyncio.Semaphore(5)
    tasks = [asyncio.create_task(call_one(client, provider, p, sem, model)) for p in prompts]
    return await asyncio.gather(*tasks)


def process_texts(samples, template, **kwargs) :

    examples_dict = kwargs["inputs"]
    texts = samples['text']
    formated_texts = []
    for text in texts :
        examples_dict["note"] = text
        new_text = template.format(**examples_dict)
        formated_texts.append(new_text)
    
    return {"questions" : formated_texts}


def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choose model, this will also be used as a file_name.")
    parser.add_argument("--provider", type=str, help="openai or claude")
    
    args = parser.parse_args()
    return args

async def main() : 
    args = parse_arguments()
    model_name = args.model
    provider = args.provider

    print("torch visible devices : ", torch.cuda.device_count(), file=sys.stderr)

    # ** whether to do finetuning from scratch 
    real_name = f"{model_name}"

    # ============================ LOAD DATASETS =============================

    # cv10, top10_dataset, filtered_notes = pd.read_pickle(DATA_PATH.joinpath("cv_processed_ranking_datasets_10fold.pkl"))
    cv10, top10_dataset, filtered_notes = load_cv_dataset()

    # cv_idx=0
    dataset = Dataset.from_pandas(filtered_notes)

    # =========================== LOAD MODEL ============================

    # model_name='mistral7b'
    # cv_idx=0
    # shot='zeroshot'
    # prompt_name='generic'
    for shot in ['zeroshot', 'fewshot'] :
        for prompt_name in ['generic', 'structured'] :
            for cv_idx, (train_idx, test_idx) in enumerate(cv10) :

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
                template = config.template(shot=shot, prompt_type=prompt_name)
                testset = dataset.filter(lambda x : x["noteid"] in test_file_ids)

                # ** processed_dataset(testset) is the test set (83 files)
                testset = testset.map(process_texts, batched=True, \
                                                fn_kwargs={"template": template, "inputs" : examples_dict})

    
                # =========================== START INFERENCE ============================

                # questions = testset['questions']
                questions = testset.to_pandas()['questions'].tolist()
                outputs = []
                
                max_new_token=200
                # provider='openai'
                # model_name='gpt-4o-mini'
                output_texts = await run_batch(questions, provider, model_name)
                # print("outputs : ", output_texts)
                
                with open(DATA_PATH.joinpath(f"{model_name}_{cv_idx}_{shot}_{prompt_name}_prediction.pkl"), 'wb') as f :
                    pickle.dump(output_texts, f)



if __name__ == "__main__"  :
    asyncio.run(main())
