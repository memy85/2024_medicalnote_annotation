import os, sys
import random
import pandas as pd
from pandas.core.common import random_state
import pandas as pd
from pandas import DataFrame
import asyncio
from openai import OpenAI, AsyncOpenAI
from utils import *

config = load_config()
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts/")

def load_dataset() -> DataFrame : 
    path = DATA_PATH.joinpath("discharge_dataset_augmented_10000.json")
    df = pd.read_json(path)
    return df

def sample_examples(jargons : DataFrame, notes : DataFrame, seed : int) :

    note_ids = notes["noteid"].unique().tolist()
    random.seed(seed)
    example_ids = random.sample(note_ids, 2)
    
    args = {}
    for i, example_id in enumerate(example_ids) : 
        
        # get note 
        example_note = notes[notes.noteid == example_id]["text"].iloc[0]

        # get answer
        answer = ""
        temp_jargon = jargons[jargons.noteid == example_id].sort_values(by="ranking")
        for _, row in temp_jargon.iterrows() :
            rank = row["ranking"]
            jargon = row["jargon"]
            answer += f"{rank}. {jargon}\n"

        args[f"note_{i+1}"] = example_note
        args[f"answer_{i+1}"] = answer
    
    return args

def prepare_prompt(prompt : str, mimic_notes : DataFrame, gold_notes : DataFrame, gold_jargons : DataFrame) -> List[str]:
    
    formatted_prompts = []
    for i, row in mimic_notes.iterrows() : 
        examples = sample_examples(gold_jargons, gold_notes, i)
        examples["context"] = row["input"]
        formatted_prompt = prompt.format(**examples)
        formatted_prompts.append(formatted_prompt)

    # formatted_prompts = formatted_prompts[:10]

    return formatted_prompts

def load_prompt() -> str :

    with open(PROMPT_PATH.joinpath("generate_annotations.txt"), 'r') as file:
        prompt = file.read()

    return  prompt


async def call_chatgpt(client, input_text: str, semaphore) -> str :
    # GPT-4 model token limit handling
    # input_chunks = input_text

    # Call the ChatGPT API for each chunk
    # for chunk in input_chunks:
    async with semaphore :
        try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": input_text},
                    ],
                    max_tokens=400,
                )
                output = response.choices[0].message.content

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            output = ""
    
    # Combine responses
    return output


async def main() :
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    sem = asyncio.Semaphore(10)

    # load dataset
    mimic_dataset = load_dataset()
    final_cv_set, final_jargons, final_notes = load_cv_dataset()

    # load prompt
    prompt = load_prompt()

    # prepare prompts
    prompts = prepare_prompt(prompt, mimic_dataset, final_notes, final_jargons)

    # call LLM to generate synthetic annotations
    results = await asyncio.gather(*[call_chatgpt(client, p, sem) for p in prompts])

    # print(results)
    # results = results * 1000
   
    # save the dataset
    mimic_dataset["raw_output"] = results
    mimic_dataset.to_json(DATA_PATH.joinpath("mad10k_raw.json"), lines=True, orient="records")

if __name__ =="__main__" :

    asyncio.run(main())


    
