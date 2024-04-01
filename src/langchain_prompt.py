
#%%
from langchain.schema import (HumanMessage, SystemMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

from utils import *
config = load_config()

#%%
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts/")

MODEL_PATH = config.model_path('mistral7b')

#%%

MAX_NEW_TOKEN = 512
top_k = 10


# gpu_llm = HuggingFacePipeline.from_model_id(
#     model_id=MODEL_PATH,
#     task="text-generation",
#     device=2,  # replace with device_map="auto" to use the accelerate library.
#     pipeline_kwargs={"max_new_tokens": 10},
# )
#%%
# llm = HuggingFaceTextGenInference(
#     inference_server_url=MODEL_PATH,
#     max_new_tokens=MAX_NEW_TOKEN,
#     top_k=50,
#     temperature=0.1,
#     repetition_penalty=1.03,
# )
#%%
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id=MODEL_PATH,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

#%%
system_prompt = '''
You are a helpful assistant, an expert in medical domain. 
Extract top 3 main diagnosis/symptoms or conditions mentioned in the medical note. 
Following the diagnosis/symptoms or conditions, identify the medical tests related to it.
If there isn't any medical tests related to it, just start listing the next important diagnosis/symptoms or conditions.
If there are no additional diagnosis/symptoms or conditions that you can identify, just list the existing ones and finalize the output. 
Don't write no symptoms, or any indication that there is no other diagnosis/symptoms or conditions.
Do not modify or abbreviate what is written in the notes. Just extract them as they are.
Make sure the highest priority is assigned with a smaller number.
We give you an example, do follow as below.
The format should be as follows

1. key symptom or condition
1.1 medical test related to 1
1.2 medical test related to 1

2. key symptom or condition
2.1 medical test related to 2

3. key symptom or condition
3.1 medical test related to 3
3.2 medical test related to 3
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

output_parser = StrOutputParser()
chain = prompt | gpu_llm | output_parser
output = chain.invoke({"input": "how can langsmith help with testing?"})
output

#%%
