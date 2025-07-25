import requests
from PIL import Image
from huggingface_hub import login
from matplotlib import pyplot as plt
import os
import json
import torch
import jsonlines
from collections import OrderedDict
from tqdm import tqdm

from lavis.models import load_model_and_preprocess


seed = 123
image_path = '/mnt3/vcr1/vcr1images'
data_path = f'/mnt/user7/Main/VisualReasoning/data/val_sample_{seed}.jsonl'

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
    # raw_image = Image.open("./data/merlion.png").convert("RGB")
    # caption = "a large fountain spewing water into the air"
    # caption = "a large"
    # caption = "a large fountain spewing"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # text_input = txt_processors["eval"](caption)
    # sample = {"image": image, "text_input": [text_input]}

    # features_multimodal = model.extract_features(sample)
    # print(features_multimodal.multimodal_embeds.shape)

    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    features_image_all = []
    features_text_all = []
    for idx in tqdm(range(len(data)), desc='Captioning..'):
        example = data[idx]
        image_fn = os.path.join(image_path, example['img_fn'])

        question = example['question']
        answer_choices = example['answer_choices']
        rationale_choices = example['rationale_choices']
        objects = example['objects']

        question = list2sentence(question, objects)

        raw_image = Image.open(image_fn).convert('RGB')

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](question)
        sample = {"image": image, "text_input": [text_input]}

        features_image = model.extract_features(sample, mode="image")
        features_text = model.extract_features(sample, mode="text")

        features_image_all.append(features_image.image_embeds_proj.detach().cpu())
        features_text_all.append(features_text.text_embeds_proj[:,0,:].unsqueeze(1).detach().cpu())

    features_image_all = torch.cat(features_image_all, dim=0)
    features_text_all = torch.cat(features_text_all, dim=0)

    z = torch.einsum('amr,bnr->abmn', features_text_all, features_image_all).squeeze(2).amax(dim=-1)

    print(z.size())

    final_data = {}
    for i in range(z.size(0)):
        data_i = z[i].tolist()
        data_i = [round(a,5) for a in data_i]
        final_data.update({str(i):data_i})

    with open("/mnt/user7/Main/VisualReasoning/results/similarity/similarity.json", "w") as f:
        json.dump(final_data, f)


    # features_image = model.extract_features(sample, mode="image")
    # features_text = model.extract_features(sample, mode="text")
    # print(features_image.image_embeds.shape)
    # # torch.Size([1, 32, 768])
    # print(features_text.text_embeds.shape)
    # # torch.Size([1, 12, 768])

    # # low-dimensional projected features
    # print(features_image.image_embeds_proj.shape)
    # # torch.Size([1, 32, 256])
    # print(features_text.text_embeds_proj.shape)
    # # torch.Size([1, 12, 256])
    # similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    # print(similarity)
    # print((features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).size())
    # # tensor([[0.3642]])