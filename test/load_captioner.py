import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset
from huggingface_hub import login
from matplotlib import pyplot as plt
import os
import json
import jsonlines
from collections import OrderedDict
from tqdm import tqdm

image_path = '/mnt3/vcr1/vcr1images'
data_path = '/mnt/user7/Main/visualreasoning/data/val_sample.jsonl'
ACCESS_TOKEN=''
login(token=ACCESS_TOKEN)

def save_fig(example, image_path, idx):
    plt.savefig(os.path.join(image_path, f'{idx}.png'))

def list2sentence(w_list, objects):
    sentence = []
    for w in w_list:
        if isinstance(w, list):
            w_ = []
            for s in w:
                obj = objects[s]
                w_.append(f'{obj}{s}')
            w_ = ' and '.join(w_)
            sentence.append(w_)
        else:
            sentence.append(w)
    sentence = ' '.join(sentence).replace(" ' ", "'").replace(" ?", "?").replace(" ,", ",").replace(" .", ".")
    return sentence

if __name__ == '__main__':
    model_id = "Salesforce/blip2-flan-t5-xxl"
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    query = 'Please describe the image content in details.'

    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    model_id_ = model_id.split('/')[-1]
    output_path = f'/mnt/user7/Main/visualreasoning/results/captions/captions_{model_id_}_vcr.jsonl'
    if os.path.exists(output_path): os.remove(output_path)

    for idx in tqdm(range(len(data)), desc='Captioning..'):
        example = data[idx]
        image_fn = os.path.join(image_path, example['img_fn'])

        question = example['question']
        answer_choices = example['answer_choices']
        rationale_choices = example['rationale_choices']
        objects = example['objects']

        question = list2sentence(question, objects)
        answer_choices = [list2sentence(a, objects) for a in answer_choices]
        rationale_choices = [list2sentence(r, objects) for r in rationale_choices]

        raw_image = Image.open(image_fn).convert('RGB')

        #save_fig(example,'/mnt/user7/Main/visualreasoning/test/images_vcr', idx)

        inputs = processor(raw_image, query, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=64)
        generated_caption = processor.decode(out[0], skip_special_tokens=True)

        print(generated_caption)

        my_data = {"question": question, 
                   "answer_choices": answer_choices, "rationale_choices": rationale_choices,
                    "generated_c": generated_caption, "image_num":example["img_id"]}
        with open(output_path, "a") as f:
            json.dump(my_data, f)
            f.write("\n")