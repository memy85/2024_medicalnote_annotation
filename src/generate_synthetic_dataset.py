import openai
import pandas as pd
from pathlib import Path
import os, sys
import time
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
SELECTED_MIMIC_DISCHARGE_PATH = DATA_PATH.joinpath("selected_mimic.pkl")


# Replace with your own OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load Prompt
prompt = config.chatgpt_prompt

# Load Mimic selected discharge notes
selected_mimic = pd.read_pickle(SELECTED_MIMIC_DISCHARGE_PATH)

def call_chatgpt(input_text, model="gpt-3.5-turbo"):
    # GPT-4 model token limit handling
    # token_limit = 30000


    # def split_text(text, max_tokens):
    #     words = text.split()
    #     chunks = []
    #     current_chunk = []
    #     current_token_count = 0

    #     for word in words:
    #         token_count = len(word) // 4  # Rough estimate of tokens per word (approx. 1 word ≈ 1.5 tokens)
    #         if current_token_count + token_count > max_tokens:
    #             chunks.append(" ".join(current_chunk))
    #             current_chunk = [word]
    #             current_token_count = token_count
    #         else:
    #             current_chunk.append(word)
    #             current_token_count += token_count

    #     if current_chunk:
    #         chunks.append(" ".join(current_chunk))

    #     return chunks

    # Split the input text into manageable chunks
    input_chunks = input_text

    # Initialize an empty response list
    responses = []

    # Call the ChatGPT API for each chunk
    # for chunk in input_chunks:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
        responses.append(response.choices[0].message['content'])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    
    # Combine responses
    return " ".join(responses)


if __name__ == "__main__" :
    

    outputs = []
    for idx, row in selected_mimic.iterrows() :

        input_text = prompt.format(context=row.text)  # Replace with your input
        response = call_chatgpt(input_text)
        print(response, file=sys.stderr)
        outputs.append(response)

        print(f"finished {idx} ====================================", file=sys.stderr)

        time.sleep(2)
        print("\n", file=sys.stderr)


        # if idx == 1 :
        #     break
    selected_mimic["gpt_output"] = outputs
    selected_mimic.to_pickle(DATA_PATH.joinpath("selected_mimic_synthetic_data.pkl"))


