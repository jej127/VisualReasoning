# VisualReasoning

This is a Pytorch implementation of BRL project: Visual Complex Reasoning.

## Setup
```
pip install -r requirements.txt
```

## Sampling 10% of VCR examples (Needs original VCR dataset)
```
python ./test/load_vcr.py
```

## Generating captions from VCR images
```
CUDA_VISIBLE_DEVICES=1 python ./test/load_captioner.py
```
## Evaluating on VCR using generated captions
```
# Evaluating Llama3-8B on VCR
CUDA_VISIBLE_DEVICES=1 python ./src/run_lm_vcr.py

# Evaluating Llama3-8B-Instruct on VCR
CUDA_VISIBLE_DEVICES=1 python ./src/run_lm_vcr_inst.py
```
