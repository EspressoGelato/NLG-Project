import os
import json
import numpy as np

from OpenFact import FactsDoc

from transformers import BartTokenizer, XLNetTokenizer
from tqdm import tqdm
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

#xlent_model_name_or_path = '/home/models/customxlnet/checkpoint-7000'
#tokenzer_xlnet = XLNetTokenizer.from_pretrained(xlnet_model_name_or_path)


FACTS_NUM = 5

block_size = 1024


def generate_template_for_bart(tgt_word_ids, kp_word_ids):
    template = [None for _ in tgt_word_ids]
    first2kp = dict()
    for k in kp_word_ids:
        if k[0] not in first2kp:
            first2kp[k[0]] = []
        first2kp[k[0]].append(k)

    ptr = 0
    while ptr < len(tgt_word_ids):
        cur_w = tgt_word_ids[ptr]
        if cur_w not in first2kp:
            ptr += 1
            continue

        candidates = first2kp[cur_w]
        found = False
        for cand in candidates:
            if len(cand) + ptr >= len(tgt_word_ids): continue
            if tgt_word_ids[ptr: ptr + len(cand)] == cand:
                found = True
                template[ptr: ptr + len(cand)] = cand
                ptr += len(cand)
                break

        if not found:
            ptr += 1
    return template


def get_data(root_path, target_path, set_type:str):
    '''
    set_type: train, valid 

    '''

    raw_examples = []
    examples = []
    docs_without_facts_counter = 0

    fout = open(f'{target_path}/refinement_{set_type}.jsonl', 'w')

    fact_file_path = os.path.join(root_path, set_type)

    for file in tqdm(sorted(os.listdir(fact_file_path))):
        path = os.path.join(fact_file_path, file)

        with open(path, 'r') as f:

            file_data = json.load(f)

        doc_id = file_data['docID']

        facts_doc = FactsDoc.Schema().load(file_data)
        if len(facts_doc.openfacts) < FACTS_NUM:
            docs_without_facts_counter +=1
            continue
        '''
        tokenized_text = tokenizer.tokenize(facts_doc.text )
        encoded_text =  tokenizer.encode("<s> " + facts_doc.text + " </s>", add_special_tokens=False, return_tensors="pt")\
                    .squeeze(0)

        bart_tokenized = tokenizer.batch_encode_plus(
                    [facts_doc.text], max_length=block_size, pad_to_max_length=True, return_tensors="pt"
                )

        prefix_tokens = [0]#[tokenizer.additional_special_tokens_ids[1]]
        ids_text_no_prefix = tokenizer.convert_tokens_to_ids(tokenized_text)
        ids_text = prefix_tokens + ids_text_no_prefix
        '''
        # strpos2index = get_strpos2index(tokenizer, encoded_text, len(prefix_tokens))
        top5salient_facts = sorted(facts_doc.openfacts, key=lambda x: x.salience)[:5]
        topfacts_sorted_by_positon = sorted(top5salient_facts, key=lambda x:x.position)

        kp_plan_str = ''

        for fact_kp in topfacts_sorted_by_positon:
            kp_plan_str = kp_plan_str + fact_kp.text + ' <s> ' 

        kp_set = []
        for fact in topfacts_sorted_by_positon:
            for tok in fact.token:
                kp_set.append(tok.word)            

        kp_set_ids = [tokenizer.encode(' ' + cur_kp, add_special_tokens=False)
                          for cur_kp in kp_set]
        kp_set_toks = [tokenizer.convert_ids_to_tokens(cur_kp) for cur_kp in kp_set_ids]            


        tgt_toks_ids = tokenizer.encode(" "+facts_doc.text.strip('<sep><cls>'), add_special_tokens=False)
        tgt_toks = tokenizer.convert_ids_to_tokens(tgt_toks_ids)
        if len(tgt_toks) > 1024:
            print(doc_id)
            continue

        template = generate_template_for_bart(tgt_toks, kp_set_toks)
        #template_ = [w if (w is not None and w != '[SEP]') else '<mask>' for w in template]
        template_ids = tokenizer.convert_tokens_to_ids(template)

        #assert len(template_ids) == len(tgt_toks_ids)

        if len(template_ids) != len(tgt_toks_ids):

            print(len(template), len(template_ids), len(tgt_toks_ids))
            #print(facts_doc.text.strip('<sep><cls>'))

        kp_set_str = ' <s> '.join(kp_set).strip()

        ret_obj = dict(
                    id=doc_id,
                    tgt=facts_doc.text.strip('<sep><cls>'),
                    prompt=topfacts_sorted_by_positon[0].text,
                    kp_plan_str=kp_plan_str.strip(),
                    kp_set_str=kp_set_str,
                    template=template,
                )
        fout.write(json.dumps(ret_obj) + '\n')
    fout.close()


if __name__ == '__main__':
    root_path = '../train-Fact2Story/'
    target_path = '../train-Fact2Story/refinement'

    get_data(root_path, target_path, set_type = 'valid')



