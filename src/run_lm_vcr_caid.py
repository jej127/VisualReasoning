import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import jsonlines
from tqdm import tqdm
import numpy as np
import random
import string
import json
import os

ACCESS_TOKEN=''
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
letters = string.ascii_uppercase
login(token=ACCESS_TOKEN)

SystemEvaluatePrompt = \
"""A chat between a curious human and an artificial intelligence assistant. You are a questioner for an image caption model and need to ask one question to get crucial information for answer prediction. Only output the question and do not provide an explanation."""

UserEvaluatePrompt = \
"""The captioner already generate a detailed image caption '{caption}'. Now you need to ask only one question for a special question '{question}' with choices '1) {choice_a} 2) {choice_b} 3) {choice_c} 4) {choice_d}'. What question you will ask? (only consider use what to set question)"""

if __name__ == '__main__':
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    data_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_fin.jsonl'
    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    output_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_caid.jsonl'
    if os.path.exists(output_path): os.remove(output_path)

    m = 10000
    for j,d in enumerate(tqdm(data)):
        if j >= m: break
        story = d["generated_c"]
        question = d["question"]
        answer_choices = d["answer_choices"]
        rationale_choices = d["rationale_choices"]
        question = d["question"]

        inputs = UserEvaluatePrompt.format(caption=story, 
                                            question=question, 
                                            choice_a=answer_choices[0],
                                            choice_b=answer_choices[1],
                                            choice_c=answer_choices[2],
                                            choice_d=answer_choices[3],)
        
        messages  = [
                        {"role": "system", "content": SystemEvaluatePrompt},
                        {"role": "user", "content": inputs},
                    ]

        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, 
                                                        add_generation_prompt=True, 
                                                        return_tensors="pt", return_dict=True).to(device)

        output = model.generate(**tokenized_chat, pad_token_id=tokenizer.eos_token_id, 
                                max_new_tokens=128)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split('\n')[-1].strip()

        d_new = d.copy()
        d_new["question_new"] = response

        with open(output_path, "a") as f:
            json.dump(d_new, f)
            f.write("\n")

        # print('#'*10)
        # print(inputs)
        # print(response)
        # print()


