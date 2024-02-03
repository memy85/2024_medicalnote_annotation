from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


llama_model_path = "/home/zhichaoyang/llm_share/llama-hf/llama-7b"
llama_2_model_path = "/data/data_user_alpha/public_models/Llama_2_hf/Llama-2-7b-hf"
llama_instruct_lora_checkpoint = "/home/htran/generation/biomed_instruct/models/llama_7b_lora/checkpoint-970"
llama_2_instruct_lora_checkpoint = "/home/htran/generation/biomed_instruct/models/llama_2_7b_all_instructions/checkpoint-975"
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
max_target_length = 512

# decide what model and checkpoint to use
model_name_or_path = llama_2_model_path
lora_checkpoint = llama_2_instruct_lora_checkpoint

# loading model original weight a
model = AutoModelForCausalLM.from_pretrained(
        llama_2_model_path,
        cache_dir=None)
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(
            llama_2_model_path,
            cache_dir=None)

tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
tokenizer.pad_token = tokenizer.eos_token

# loading lora checkpoint
model.enable_input_require_grads()
model = PeftModel.from_pretrained(model, lora_checkpoint)

# loading prompt, see some examples at /home/htran/generation/biomed_instruct/prompts
prompt = open("prompts/mediqa_taska_prompt.txt").read()  # one shot prompt

# loading your task input, for example a Dialogue
task_input = "Doctor: When did your pain begin? \nPatient: I've had low back pain for about eight years now.\nDoctor: Is there any injury? \n Patient: Yeah, it started when I fell in an A B C store.\nDoctor: How old are you now?\nPatient: I'm twenty six.  \nDoctor: What kind of treatments have you had for this low back pain? \nPatient: Yeah, I got referred to P T, and I went, but only once or twice, um, and if I remember right, they only did the electrical stimulation, and heat. \nDoctor: I see, how has your pain progressed over the last eight years? \nPatient: It's been pretty continuous, but it's been at varying degrees, sometimes are better than others. \nDoctor: Do you have any children? \nPatient: Yes, I had my son in August of two thousand eight, and I've had back pain since giving birth. \nDoctor: Have you had any falls since the initial one? \nPatient: Yes, I fell four or five days ago while I was mopping the floor. \nDoctor: Did you land on your lower back again?\nPatient: Yes, right onto my tailbone. \nDoctor: Did that make the low back pain worse? \nPatient: Yes. \nDoctor: Have you seen any other doctors for this issue? \nPatient: Yes, I saw Doctor X on January tenth two thousand nine, and I have a follow up appointment scheduled for February tenth two thousand nine."
dict = {"dialogue": task_input}

# combine with prompt for actual input
input = prompt.format(**dict)

# tokenize
input_ids = tokenizer(input, return_tensors="pt").input_ids.cuda()

#generate with max length
output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_target_length)
arr_output = output_ids.detach().cpu().numpy()

# find the index of generation text, default to the length of input
start_of_generate_index = input_ids.shape[1]
pred_output = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
print(pred_output)




