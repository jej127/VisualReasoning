import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset
from huggingface_hub import login
import jsonlines
import os
import numpy as np
import json

#123
#42
#12
#111
#222
#333
#444
seed=444
np.random.seed(seed)

if __name__ == '__main__':
    image_path = '/mnt3/vcr1/vcr1images'
    data_path = '/mnt3/vcr1/val.jsonl'
    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    # for i in range(5):
    #     print(data[i])
    #     print(data[i]['img_fn'])
    #     print()

    idxs = np.random.choice(len(data), 2653, replace=False)
    output_path = f'/mnt/user7/Main/VisualReasoning/data/val_sample_{seed}.jsonl'
    if os.path.exists(output_path): os.remove(output_path)
    for idx in idxs:
        with open(output_path, "a") as f:
            json.dump(data[idx], f)
            f.write("\n")