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


if __name__ == '__main__':
    raw_image = Image.open("./data/merlion.png").convert("RGB")
    caption = "a large fountain spewing water into the air"
    caption = "a large"
    caption = "a large fountain spewing"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)
    #image = torch.cat([image], dim=0)
    sample = {"image": image, "text_input": [text_input]}

    

    features_multimodal = model.extract_features(sample)
    print(features_multimodal.multimodal_embeds.shape)

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