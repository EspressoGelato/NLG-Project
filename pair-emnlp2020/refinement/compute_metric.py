import json
import os
from transformers import BartTokenizer

from torchtext.data.metrics import bleu_score
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

from nltk.translate import bleu_score

from nltk.translate.bleu_score import SmoothingFunction
chencherry = SmoothingFunction()


gt_path = '../data-fact/refinement/fact/valid.jsonl'
pred_path = './output/fact/valid-fact-pos-pair.jsonl'#valid-pair-gt-template.jsonl'#test-fact-pos-pair.jsonl'


gt_data = []
with open(gt_path, 'r') as f:
    for ln in f:
        gt_data.append(json.loads(ln))

pred_data = []
with open(pred_path, 'r') as f:

    for ln in f:
        pred_data.append(json.loads(ln))


pred_list = []
reference_list = []


for i, pred in enumerate(pred_data):
    pred_list.append(tokenizer.tokenize(pred['output_str']))
    reference_list.append([tokenizer.tokenize(gt_data[i]['tgt'])])
    
    print(len(pred_list), pred_data[i]['output_str'])
    print('#'*100)
    print(len(reference_list), gt_data[i]['tgt'])
score = bleu_score.corpus_bleu(reference_list, pred_list, smoothing_function=chencherry.method2)
print("corpus bleu:", score)
for pred, ref in zip(pred_list, reference_list):
    score = bleu_score.sentence_bleu(ref, pred, smoothing_function=chencherry.method2)
    print(score)
