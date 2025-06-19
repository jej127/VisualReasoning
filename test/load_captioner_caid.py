import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
from datasets import load_dataset
from huggingface_hub import login
from matplotlib import pyplot as plt
import os
import json
import jsonlines
from collections import OrderedDict
from tqdm import tqdm
import torch
from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.utils import get_balanced_memory
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

image_path = '/mnt3/vcr1/vcr1images'
data_path = '/mnt/user7/Main/VisualReasoning/data/val_sample.jsonl'
data_path2 = '/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_caid.jsonl'
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

    query_old = 'Please describe the image content in details.'
    query = """Please answer the question in a complete sentence, not in words. Do not omit any key information. If the answer is either 'yes' or 'no', add the supporting sentence."""

    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    data2 = []
    with jsonlines.open(data_path2) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data2.append(d)

    model_id_ = model_id.split('/')[-1]
    output_path = f'/mnt/user7/Main/VisualReasoning/results/captions/captions_{model_id_}_vcr_fin_caid.jsonl'
    if os.path.exists(output_path): os.remove(output_path)
# {"question": "Is it preparing to rain on person0 and person1 and person2?", "answer_choices": ["No, person0 and person1 and person2 are not outside.", "No it is not.", "Yes, it is likely that it is preparing to rain on person0 and person1 and person2.", "Yes, it is snowing outside."], "rationale_choices": ["person0 and person1 and person2 are wearing raincoats.", "Highly possible with person0 and person1 and person2 carrying the umbrella and sky looking a bit cloudy.", "Moving air around with a fan makes a breeze that cools people off.", "The sky is filled with clouds, clouds precede rain therefore it is likely that it is preparing to rain."], "generated_c": "a man in a vest and tie talking to two men", "image_num": "val-7701", "question_new": "Is the scene taking place outdoors?"}

    for idx in tqdm(range(len(data)), desc='Captioning..'):
        example = data[idx]
        example2 = data2[idx]
        image_fn = os.path.join(image_path, example['img_fn'])

        question = example2['question']
        inital_gen = example2["generated_c"]
        answer_choices = example2['answer_choices']
        rationale_choices = example2['rationale_choices']
        question_new = example2['question_new']

        raw_image = Image.open(image_fn).convert('RGB')

        #save_fig(example,'/mnt/user7/Main/VisualReasoning/test/images_vcr', idx)

        # input_prompt = f'{question_new} {query}'
        input_prompt = f'{inital_gen}. {question_new} {query}'

        inputs = processor(raw_image, input_prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=64)
        generated_caption = processor.decode(out[0], skip_special_tokens=True)

        print(generated_caption)

        my_data = example2.copy()
        my_data["generated_new_c"] = generated_caption

        with open(output_path, "a") as f:
            json.dump(my_data, f)
            f.write("\n")