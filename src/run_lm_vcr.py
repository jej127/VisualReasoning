import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import jsonlines
from tqdm import tqdm
import numpy as np
import random
import string

ACCESS_TOKEN=''
model_id = "meta-llama/Meta-Llama-3-8B"
letters = string.ascii_lowercase
login(token=ACCESS_TOKEN)

SystemEvaluatePrompt = \
"""Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
Note:
(1) Please only output the most likely answer index based on the given information, and do not output any other content."""

SystemEvaluatePrompt_rat = \
"""Below is a multiple-choice question with a story, an answer for the question, and several rationale options. Based on the content of the story and the given question and the answer, please infer the most likely rationale that supports the answer and output the rationale index.
Note:
(1) Please only output the most likely rationale index based on the given information, and do not output any other content."""

UserEvaluatePrompt4Choices = \
"""Context:
{story}

Question:
{question}

Candidate Answers:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

UserEvaluatePrompt4Choices_rat = \
"""Context:
{story}

Question:
{question}

Answer:
{answer}

Candidate Rationales:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Rationale:"""

def process_output(pred, choices):
    try:
        pred = pred.lower().replace("（", "(").replace("）", ")").replace(".", "")
        choices = [
            choice.replace(" & ", " and ")
            for choice in choices
        ]
        lines = pred.split("\n")
        for j in range(len(lines)):
            output = lines[j]
            if output:
                alphabets = {
                    "normal": [
                        f"({letters[i]})" for i in range(4)
                    ],
                    "paranthese": [
                        f"[{letters[i]}]" for i in range(4)
                    ],
                    "dot": [f": {letters[i]}" for i in range(4)],
                    "option": [
                        f"option {letters[i]}" for i in range(4)
                    ],
                    "option1": [
                        f"option ({letters[i]})"
                        for i in range(4)
                    ],
                    "choice": [
                        f"choice {letters[i]}" for i in range(4)
                    ],
                    "choice1": [
                        f"choice ({letters[i]})"
                        for i in range(4)
                    ],
                    "选项": [
                        f"选项 {letters[i]}" for i in range(4)
                    ],
                    "选项1": [
                        f"选项 ({letters[i]})" for i in range(4)
                    ],
                }

                for v in alphabets.values():
                    for a in v:
                        if a in output:
                            return v.index(a)
                for c in choices:
                    if c.lower() in output:
                        return choices.index(c)
                if len(output.strip()) == 1 and output in letters[:4]:
                    return letters.index(output)
                if len(output.strip()) == 1 and output in ['1','2','3','4']:
                    return ['1','2','3','4'].index(output)
                if output[0] in letters[:4] and output[1] in [
                    "<",
                    "[",
                    "(",
                    ")",
                    ":",
                ]:
                    return letters.index(output[0])
    except Exception as e:
        print("Error in processing output", type(e).__name__, "–", e)

    return -1

if __name__ == '__main__':
    # pipeline = transformers.pipeline(
    # "text-generation",
    # model=model_id,
    # model_kwargs={"torch_dtype": torch.float16},
    # device_map="auto",
    # return_full_text=False
    # )
    #result = pipeline("Hey how are you doing today?")[0]['generated_text']

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    data_path = '/mnt/user7/Main/visualreasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_fin.jsonl'
    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    answer_path = '/mnt/user7/Main/visualreasoning/data/val_sample.jsonl'
    gt_ans,gt_rat = [],[]
    with jsonlines.open(answer_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            gt_ans.append(d["answer_label"])
            gt_rat.append(d["rationale_label"])

    gt_ans, gt_rat = np.array(gt_ans), np.array(gt_rat)

    pred_ans, pred_rat = [], []
    m = 10000
    for j,d in enumerate(tqdm(data)):
        if j >= m: break
        story = d["generated_c"]
        question = d["question"]
        answer_choices = d["answer_choices"]
        rationale_choices = d["rationale_choices"]
        question = d["question"]
        ans_idx = gt_ans[j]
        answer = answer_choices[ans_idx]

        input_q2a = UserEvaluatePrompt4Choices.format(story=story, 
                                                      question=question, 
                                                      choice_a=answer_choices[0],
                                                      choice_b=answer_choices[1],
                                                      choice_c=answer_choices[2],
                                                      choice_d=answer_choices[3],)
        input_qa2r = UserEvaluatePrompt4Choices_rat.format(story=story, 
                                                           question=question, 
                                                           answer=answer,
                                                           choice_a=rationale_choices[0],
                                                           choice_b=rationale_choices[1],
                                                           choice_c=rationale_choices[2],
                                                           choice_d=rationale_choices[3])
        
        input_ids_q2a = tokenizer(SystemEvaluatePrompt+'\n\n'+input_q2a, return_tensors="pt").to(device)
        input_ids_qa2r = tokenizer(SystemEvaluatePrompt_rat+'\n\n'+input_qa2r, return_tensors="pt").to(device)

        prompt_length_q2a = input_ids_q2a['input_ids'].shape[1]
        prompt_length_qa2r = input_ids_qa2r['input_ids'].shape[1]

        output_q2a = model.generate(**input_ids_q2a, pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=32)
        output_qa2r = model.generate(**input_ids_qa2r, pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=32)

        response_q2a = tokenizer.decode(output_q2a[0][prompt_length_q2a:], skip_special_tokens=True).strip()
        response_qa2r = tokenizer.decode(output_qa2r[0][prompt_length_qa2r:], skip_special_tokens=True).strip()

        r_q2a = process_output(response_q2a, answer_choices)
        r_qa2r = process_output(response_qa2r, rationale_choices)

        pred_ans.append(r_q2a)
        pred_rat.append(r_qa2r)

        # print('#'*10)
        # print(input_q2a)
        # print(response_q2a)
        # print(r_q2a)
        # print()
        # print(input_qa2r)
        # print(response_qa2r)
        # print(r_qa2r)
        # print()

    pred_ans, pred_rat = np.array(pred_ans), np.array(pred_rat)

    gt_ans, gt_rat = gt_ans[:m], gt_rat[:m]

    # print(pred_ans.shape)
    # print(gt_ans.shape)
    # print(pred_rat.shape)
    # print(gt_rat.shape)

    tf_ans = (pred_ans == gt_ans)*1
    tf_rat = (pred_rat == gt_rat)*1

    acc_q2a = np.mean(tf_ans)*100
    acc_qa2r = np.mean(tf_rat)*100
    acc_q2ar = np.mean(tf_ans*tf_rat)*100

    print(f'Q2A: {round(acc_q2a,1)}')
    print(f'QA2R: {round(acc_qa2r,1)}')
    print(f'Q2AR: {round(acc_q2ar,1)}')