import transformers
import torch
from huggingface_hub import login
import jsonlines
from tqdm import tqdm
import numpy as np
import random

ACCESS_TOKEN='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'
model_id = "meta-llama/Meta-Llama-3-8B"
login(token=ACCESS_TOKEN)

prompt_i2c = \
"""Question: the sentence "{image}" is relevant to
(1) {caption0}
(2) {caption1}

Answer:"""

prompt_c2i = \
"""Question: the sentence "{caption}" is relevant to
(1) {image0}
(2) {image1}

Answer:"""

if __name__ == '__main__':
    pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
    return_full_text=False
    )
    #result = pipeline("Hey how are you doing today?")[0]['generated_text']

    data_path = '/mnt/user7/Main/visualreasoning/results/captions/captions_blip2-flan-t5-xxl.jsonl'
    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    scores = []
    for d in tqdm(data):
        original_c0, original_c1 = d['original_c0'], d['original_c1']
        generated_c0, generated_c1 = d['generated_c0'], d['generated_c1']

        input_text1_i2c = prompt_i2c.format(image=generated_c0, caption0=original_c0, caption1=original_c1)
        input_text2_i2c = prompt_i2c.format(image=generated_c1, caption0=original_c0, caption1=original_c1)


        result1_i2c = pipeline(input_text1_i2c, max_new_tokens=64,
                               pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
        result2_i2c = pipeline(input_text2_i2c, max_new_tokens=64,
                               pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']

        if '(1)' in result1_i2c and '(2)' not in result1_i2c:
            ans1_i2c = 1
        elif '(2)' in result1_i2c and '(1)' not in result1_i2c:
            ans1_i2c = 0
        else:
            ans1_i2c = (random.random() < 0.5)*1

        if '(2)' in result2_i2c and '(1)' not in result2_i2c:
            ans2_i2c = 1
        elif '(1)' in result2_i2c and '(2)' not in result2_i2c:
            ans2_i2c = 0
        else:
            ans2_i2c = (random.random() < 0.5)*1

        ans_i2c = ans1_i2c * ans2_i2c

        ##################################

        input_text1_c2i = prompt_c2i.format(caption=original_c0, image0=generated_c0, image1=generated_c1)
        input_text2_c2i = prompt_c2i.format(caption=original_c1, image0=generated_c0, image1=generated_c1)

        result1_c2i = pipeline(input_text1_c2i, max_new_tokens=64,
                               pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
        result2_c2i = pipeline(input_text2_c2i, max_new_tokens=64,
                               pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
    

        if '(1)' in result1_c2i and '(2)' not in result1_c2i:
            ans1_c2i = 1
        elif '(2)' in result1_c2i and '(1)' not in result1_c2i:
            ans1_c2i = 0
        else:
            #ans1_c2i = (random.random() < 0.5)*1
            ans1_c2i = (random.random() < 0.5)*1

        if '(2)' in result2_c2i and '(1)' not in result2_c2i:
            ans2_c2i = 1
        elif '(1)' in result2_c2i and '(2)' not in result2_c2i:
            ans2_c2i = 0
        else:
            ans2_c2i = (random.random() < 0.5)*1

        ans_c2i = ans1_c2i * ans2_c2i

        scores.append([ans_i2c, ans_c2i, ans_i2c*ans_c2i])

    scores = np.array(scores)
    print('Text score')
    print(np.mean(scores[:,0])*100)
    print('Image score')
    print(np.mean(scores[:,1])*100)
    print('Group score')
    print(np.mean(scores[:,2])*100)