import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import jsonlines
from tqdm import tqdm
import numpy as np
import random
import string
import os
import json
import bm25s
import Stemmer
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
seed=123

ACCESS_TOKEN=''
model_id = "meta-llama/Meta-Llama-3-8B"
letters = string.ascii_lowercase
login(token=ACCESS_TOKEN)

Head = ""

# SystemEvaluatePrompt = \
# """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer. Keep your answer concise, one sentence is enough. You must choose one of the above answer options.

# Examples:
# """

# SystemEvaluatePrompt_rat = \
# """Below is a multiple-choice question with a story, an answer for the question, and several rationale options. Based on the content of the story and the given question and the answer, please infer the most likely rationale that supports the answer. Keep your answer concise, one sentence is enough. You must choose one of the above rationale options.

# Examples:
# """

# SystemEvaluatePrompt = \
# """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]" as the answer index.
# """

# SystemEvaluatePrompt_rat = \
# """Below is a multiple-choice question with a story, an answer for the question, and several rationale options. Based on the content of the story and the given question and the answer, please infer the most likely rationale that supports the answer and output one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]" as the rationale index.
# """

# SystemEvaluatePrompt = \
# """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer. Output only one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]". Do not provide an explanation.
# """

# SystemEvaluatePrompt_rat = \
# """Below is a multiple-choice question with a story, an answer for the question, and several rationale options. Based on the content of the story and the given question and the answer, please infer the most likely rationale that supports the answer. Output only one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]". Do not provide an explanation.
# """

# SystemEvaluatePrompt = \
# """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
# """

# SystemEvaluatePrompt_rat = \
# """Below is a multiple-choice question with a story, an answer for the question, and several rationale options. Based on the content of the story and the given question and the answer, please infer the most likely rationale that supports the answer and output the rationale index.
# """

# SystemEvaluatePrompt = \
# """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer.
# """

# SystemEvaluatePrompt_rat = \
# """Below is a multiple-choice question with a story, an answer for the question, and several rationale options. Based on the content of the story and the given question and the answer, please infer the most likely rationale that supports the answer.
# """

UserEvaluatePrompt4Choices = \
"""Context:
{story}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

UserEvaluatePrompt4Choices_rat = \
"""Context:
{story}

Question:
Why is the statement "{answer}" the answer to the question "{question}"

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

def process_output(pred, choices):
    try:
        # print('#'*10)
        # print(pred)
        pred = pred.lower().replace("（", "(").replace("）", ")").replace(".", "")
        choices = [
            choice.replace(" & ", " and ")
            for choice in choices
        ]
        lines = pred.split("\n")
        for j in range(len(lines)):
            output = lines[j]
            # print('@'*10)
            # print(output)
            if output:
                alphabets = {
                    "normal": [
                        f"({letters[i]})" for i in range(4)
                    ],
                    "paranthese": [
                        f"[{letters[i]}]" for i in range(4)
                    ],
                    "paranthese2": [
                        f"{letters[i]})" for i in range(4)
                    ],
                    "dot": [f": {letters[i]}" for i in range(4)],
                    "dot2": [f"{letters[i]}. " for i in range(4)],
                    "dot3": [f":{letters[i]}" for i in range(4)],
                    "option": [
                        f"option {letters[i]}" for i in range(4)
                    ],
                    "option1": [
                        f"option ({letters[i]})"
                        for i in range(4)
                    ],
                    "option2": [
                        f"{letters[i]} is"
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
    pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    return_full_text=False
    )
    #result = pipeline("Hey how are you doing today?")[0]['generated_text']

    #data_path = '/mnt/user7/Main/VisualReasoning/results/captions/captions_Molmo-7B-D-0924_vcr_molmo.jsonl'
    data_path = f'/mnt/user7/Main/VisualReasoning/results/captions/captions_blip2-flan-t5-xxl_vcr_fin_caid_{seed}_2.jsonl'
    data = []
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    answer_path = f'/mnt/user7/Main/VisualReasoning/data/val_sample_{seed}.jsonl'
    gt_ans,gt_rat = [],[]
    with jsonlines.open(answer_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            gt_ans.append(d["answer_label"])
            gt_rat.append(d["rationale_label"])

    gt_ans, gt_rat = np.array(gt_ans), np.array(gt_rat)
    model_id_ = model_id.split('/')[-1]
    ic_path = f'/mnt/user7/Main/VisualReasoning/results/reasoning/{model_id_}_reasoning_caid_{seed}.jsonl'
    data_ic = []
    with jsonlines.open(ic_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            story, question = d['story'].strip(), d['question']
            answer_choices, rationale_choices, answer = d["answer_choices"], d["rationale_choices"], d['answer']
            #a_fin, r_fin = d['a'], d['r']
            a_fin, r_fin = d['a'].replace('[','').replace(']',''), d['r'].replace('[','').replace(']','')
            p_q2a = UserEvaluatePrompt4Choices.format(story=story, 
                                                        question=question, 
                                                        choice_a=answer_choices[0],
                                                        choice_b=answer_choices[1],
                                                        choice_c=answer_choices[2],
                                                        choice_d=answer_choices[3],)
            p_qa2r = UserEvaluatePrompt4Choices_rat.format(story=story, 
                                                            question=question, 
                                                            answer=answer,
                                                            choice_a=rationale_choices[0],
                                                            choice_b=rationale_choices[1],
                                                            choice_c=rationale_choices[2],
                                                            choice_d=rationale_choices[3])
            data_ic.append({'a':p_q2a+a_fin.replace('\n',' '), 'r':p_qa2r+r_fin.replace('\n',' ')})

    answer_corpus = [d['a'] for d in data_ic]
    rationale_corpus = [d['r'] for d in data_ic]

    stemmer = Stemmer.Stemmer("english")
    answer_corpus_tokens = bm25s.tokenize(answer_corpus, stopwords="en", stemmer=stemmer)
    rationale_corpus_tokens = bm25s.tokenize(rationale_corpus, stopwords="en", stemmer=stemmer)

    answer_retriever = bm25s.BM25()
    answer_retriever.index(answer_corpus_tokens)
    rationale_retriever = bm25s.BM25()
    rationale_retriever.index(rationale_corpus_tokens)

    reasoning_path = '/mnt/user7/Main/VisualReasoning/results/reasoning'
    #model_id_ = model_id.split('/')[-1]
    reasoning_output_path = os.path.join(reasoning_path, f'{model_id_}_reasoning_ic_{seed}_.txt')
    if os.path.exists(reasoning_output_path): os.remove(reasoning_output_path)

    output_path = os.path.join(reasoning_path, f'{model_id_}_reasoning_ic_{seed}_.jsonl')
    if os.path.exists(output_path): os.remove(output_path)

    similarity_path = '/mnt/user7/Main/VisualReasoning/results/similarity/similarity.json'
    with open(similarity_path) as json_file:
        similarity_data = json.load(json_file)

    pred_ans, pred_rat = [], []
    m = 10000
    for j,d in enumerate(tqdm(data)):
        if j >= m: break
        story = d["generated_c"]
        question = d["question"]
        answer_choices = d["answer_choices"]
        rationale_choices = d["rationale_choices"]
        question_new = d["question_new"]
        story_new = d["generated_new_c2"]
        img_id = d['image_num']

        # if len(answer_new.strip().split()) > 1:
        #     answer_new = answer_new[0].upper() + answer_new[1:] + '.'
        #     story = story + '. ' + answer_new
        if len(story_new.strip().split()) > 1:
            story = f'{story}. {story_new}'

        ans_idx = gt_ans[j]
        rat_idx = gt_rat[j]
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
        

        q2a_tokens = bm25s.tokenize(input_q2a, stemmer=stemmer)
        qa2r_tokens = bm25s.tokenize(input_qa2r, stemmer=stemmer)
        # corpus=answer_corpus
        # corpus=rationale_corpus
        results_q2a, scores_q2a = answer_retriever.retrieve(q2a_tokens, k=16)
        results_qa2r, scores_qa2r = rationale_retriever.retrieve(qa2r_tokens, k=16)


        id2score_q2a, id2score_qa2r = [],[]
        multimodal_scores = similarity_data[str(j)]
        for i in range(results_q2a.shape[1]):
            idx_q2a, score_q2a = results_q2a[0, i], scores_q2a[0, i]
            idx_qa2r, score_qa2r = results_qa2r[0, i], scores_qa2r[0, i]

            doc_q2a = answer_corpus[idx_q2a]
            doc_qa2r = rationale_corpus[idx_qa2r]

            if question in doc_q2a: continue

            multimodal_score_q2a = multimodal_scores[idx_q2a]
            multimodal_score_qa2r = multimodal_scores[idx_qa2r]

            id2score_q2a.append((idx_q2a,0.5*score_q2a+multimodal_score_q2a))
            id2score_qa2r.append((idx_qa2r,0.5*score_qa2r+multimodal_score_qa2r))

        id2score_q2a.sort(key=lambda x: x[1], reverse=True)
        id2score_qa2r.sort(key=lambda x: x[1], reverse=True)


        ic_q2a, ic_qa2r = '',''
        num_example = 0
        for k in range(10):
            idx_q2a,idx_qa2r = id2score_q2a[k][0], id2score_qa2r[k][0]

            doc_q2a = answer_corpus[idx_q2a]
            doc_qa2r = rationale_corpus[idx_qa2r]

            ic_q2a += f'{doc_q2a}\n\n'
            ic_qa2r += f'{doc_qa2r}\n\n'

            num_example += 1
            if num_example == 4: break

        try:
            input_ids_q2a = pipeline(Head+ic_q2a+input_q2a, max_new_tokens=16, 
                                    pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
            input_ids_qa2r = pipeline(Head+ic_qa2r+input_qa2r, max_new_tokens=16,
                                    pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']

            response_q2a = input_ids_q2a
            response_qa2r = input_ids_qa2r

            r_q2a = process_output(response_q2a.strip(), answer_choices)
            r_qa2r = process_output(response_qa2r.strip(), rationale_choices)
        except torch.cuda.OutOfMemoryError:
            response_q2a,response_qa2r ='',''
            r_q2a, r_qa2r = 0,0

        #ic_q2a,ic_qa2r='',''
        # input_ids_q2a = pipeline(Head+ic_q2a+input_q2a, max_new_tokens=16, 
        #                         pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
        # input_ids_qa2r = pipeline(Head+ic_qa2r+input_qa2r, max_new_tokens=16,
        #                         pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']

        # response_q2a = input_ids_q2a
        # response_qa2r = input_ids_qa2r

        # r_q2a = process_output(response_q2a.strip(), answer_choices)
        # r_qa2r = process_output(response_qa2r.strip(), rationale_choices)

        # print('#'*10)
        # print(response_q2a)
        # print(r_q2a)

        if r_q2a not in [0,1,2,3]:
            r_q2a = random.randrange(4)
        if r_qa2r not in [0,1,2,3]:
            r_qa2r = random.randrange(4)

        pred_ans.append(r_q2a)
        pred_rat.append(r_qa2r)

        qa_t, qar_t = str(r_q2a==gt_ans[j]), str(r_qa2r==gt_rat[j])
        a,r = ['A','B','C','D'][ans_idx], ['A','B','C','D'][rat_idx]
        with open(reasoning_output_path, "a") as f:
            f.write("#"*20 + "\n")
            f.write(img_id + "\n")
            f.write("#"*10 + 'Q2A' + "\n")
            f.write("<Prompt>\n")
            f.write(Head+ic_q2a+input_qa2r+"\n\n")
            f.write("<Output>\n")
            f.write(response_q2a + "\n\n")
            f.write(f"GT: {a}\n")
            f.write(f"Result: {qa_t}\n\n")
            f.write("#"*10 + 'QA2R' + "\n")
            f.write("<Prompt>\n")
            f.write(Head+ic_qa2r+input_qa2r+"\n\n")
            f.write("<Output>\n")
            f.write(response_qa2r + "\n\n")
            f.write(f"GT: {r}\n")
            f.write(f"Result: {qar_t}\n\n")

        final_a = response_q2a.split('assistant\n')[-1]
        final_r = response_qa2r.split('assistant\n')[-1]
        my_data = {"story": story, "question": question, 
                   "answer_choices": answer_choices, "rationale_choices": rationale_choices, "answer": answer,
                    "a": final_a, "r": final_r, "image_num":img_id}
        with open(output_path, "a") as f:
            json.dump(my_data, f)
            f.write("\n")

        # print('#'*10)
        # print(input_q2a)
        # print(response_q2a)
        # print(r_q2a)
        # print(ans_idx)
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