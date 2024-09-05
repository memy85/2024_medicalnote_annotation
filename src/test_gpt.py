from openai import OpenAI

client = OpenAI()

#%% load prompts



#%%

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a professor in mathematics"},
        {"role": "user", "content": "What is 2 times 2?"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
