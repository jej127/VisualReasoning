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
#model_id = "meta-llama/Llama-3.1-8B-Instruct"
letters = string.ascii_uppercase
login(token=ACCESS_TOKEN)

SystemEvaluatePrompt = \
"""A chat between a curious human and an artificial intelligence assistant. You are a summarizer for an image caption model and need to summarize a question and and its answer into a single sentence. Only output the summarized sentence and do not provide an explanation."""

UserEvaluatePrompt = \
"""Examples)
Question: What is the emotion shown on person2's face in the image?
Answer: happy
Summarize: The emotion shown on person2's face in the image is happiness.

Question: What is the relationship between the two people in the image?
Answer: friends
Summarize: The two people in the image are friends.

Question: Is there another person present in the theater besides the man watching the movie?
Answer: yes
Summarize: There is another person present in the theater besides the man watching the movie.

Question: Is the weapon in the image held by one of the children?
Answer: no
Summarize: The weapon in the image is not held by one of the children.

Question: {question}
Answer: {answer}
Summarize: """

if __name__ == '__main__':
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    #data_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_fin.jsonl'
    #data_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_Molmo-7B-D-0924_vcr_123.jsonl'
    data_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_fin_caid_123.jsonl'
    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    # output_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_caid.jsonl'
    #output_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_Molmo-7B-D-0924_vcr_123_caid.jsonl'
    output_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_fin_caid_123_2.jsonl'
    if os.path.exists(output_path): os.remove(output_path)

    m = 10000
    for j,d in enumerate(tqdm(data)):
        if j >= m: break
        story = d["generated_c"]
        question = d["question"]
        answer_choices = d["answer_choices"]
        rationale_choices = d["rationale_choices"]
        question = d["question_new"]
        answer = d["generated_new_c"]

        inputs = UserEvaluatePrompt.format(question=question, 
                                            answer=answer)
        
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
        d_new["generated_new_c2"] = response

        with open(output_path, "a") as f:
            json.dump(d_new, f)
            f.write("\n")

        # print('#'*10)
        # print(inputs)
        # print(response)
        # print()


