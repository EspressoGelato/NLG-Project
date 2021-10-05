"""Code to build opinon and news data from The New York Times corpus"""
import os
import argparse
import utils
import json
import time
import numpy as np
from tqdm import tqdm

from utils import (
    get_chunks,
    chunk_tagged_sents,
    _process_kp_list,
    detokenize_stanford,
    fuzzy_match_kp,
    calculate_kp_offsets,
    extract_body,
    get_nytimes_topic_signatures,
    generate_template_for_bart,
)


def load_nytimes_ids(domain, set_type):
    """Load sample ids for train/dev/test"""

    path = f'{domain}/{set_type}.ids'
    ids = []
    for ln in open(path):
        ids.append(ln.strip())
    return ids


def load_nytimes_rawdata(domain, ids):
    """Load raw data, note that the raw data has to be purchased from LDC:
        `The New York Times Annotated Corpus`
        https://catalog.ldc.upenn.edu/LDC2008T19

    After uncompress the data, we assume the data is stored in files with
    the naming convention `nyt_{year}.jsonl`, where `year` is 1987-2007
    """
    raw_data = []
    for year in range(1987, 2008):
        path = f'nytimes_raw/nyt_{year}.jsonl'
        if not os.path.exists(path):
            continue
        for ln in open(path):
            cur_obj = json.loads(ln)
            cur_id = cur_obj['file_id']
            if cur_id not in ids:
                continue

            body_str, _ = extract_body(cur_obj['full_article'], domain)

            ret_obj = dict(
                    id=cur_obj['file_id'],
                    prompt=cur_obj['headline'],
                    tgt=body_str)
            raw_data.append(ret_obj)
    return raw_data

def run_corenlp(raw_data, domain, set_type, save_to_disk=True):
    """Run StanfordCoreNLP pipeline (tokenize,ssplit,pos)

    If the results is already in the disk, load without running.
    """
    output_path = f'{domain}/{set_type}.corenlp.jsonl'
    if os.path.exists(output_path):
        corenlp_results = [json.loads(ln) for ln in open(output_path)]
        if len(corenlp_results) == len(raw_data):
            print('corenlp results loaded')
            return corenlp_results

    from pycorenlp import StanfordCoreNLP

    # assuming stanford corenlp server running at 9000
    nlp = StanfordCoreNLP("http://localhost:9000")
    if save_to_disk:
        assert domain
        assert set_type
        fout = open(output_path, 'w')

    corenlp_results = []
    for ln in tqdm(raw_data):
        tgt = ln['tgt']
        proc = nlp.annotate(tgt,
                            properties={'annotators': 'tokenize,ssplit,pos',
                                        'outputFormat': 'json'})

        ret_obj = dict(
                id=ln['id'],
                prompt=ln['prompt'],
                proc=proc)

        if save_to_disk:
            fout.write(json.dumps(ret_obj) + '\n')
        corenlp_results.append(ret_obj)

    if save_to_disk:
        fout.close()
    print('corenlp results loaded')
    return corenlp_results

def run_chunking(corenlp_results, domain, set_type, save_to_disk=True):
    """Run regex based chunking for NP and VP.
    """
    output_path = f'{domain}/{set_type}.chunk.jsonl'
    if os.path.exists(output_path):
        chunk_results = [json.loads(ln) for ln in open(output_path)]
        if len(chunk_results) == len(corenlp_results):
            print('chunking results loaded')
            return chunk_results

    if save_to_disk:
        fout = open(output_path, 'w')

    chunk_results = []
    for ln in tqdm(corenlp_results):
        np_list, vp_list = [], []
        tgt_proc = ln['proc']['sentences']
        for sent in tgt_proc:
            cur_tokens = sent['tokens']
            cur_pos = [(tok['originalText'], tok['pos']) for tok in cur_tokens]
            cur_np_ = get_chunks(chunk_tagged_sents(cur_pos), 'NP')
            cur_vp_ = get_chunks(chunk_tagged_sents(cur_pos), 'VP')
            np_list.append(cur_np_)
            vp_list.append(cur_vp_)
        ret_obj = dict(
                id=ln['id'],
                np_list=np_list,
                vp_list=vp_list
                )
        chunk_results.append(ret_obj)
        if save_to_disk:
            fout.write(json.dumps(ret_obj) + '\n')
    print('chunking results saved')
    print('chunking results loaded')
    return chunk_results



def generate_kp(raw_data, domain, set_type, cache_intermediate):
    """Generate keyphrase from raw text, need to run StanfordCoreNLP first,
    then regex based chunking
    """
    output_path = f'{domain}/{set_type}.kp.jsonl'
    corenlp_results = run_corenlp(raw_data, domain, set_type, cache_intermediate)
    if os.path.exists(output_path):
        kp_data = [json.loads(ln) for ln in open(output_path)]
        print('kp data loaded')
        return kp_data, corenlp_results

    chunk_results = run_chunking(corenlp_results, domain, set_type, cache_intermediate)
    topic_sig = get_nytimes_topic_signatures(domain)
    kp_data = []
    if cache_intermediate:
        fout = open(output_path, 'w')

    for ln in tqdm(chunk_results):
        cur_ts = topic_sig[ln['id']]

        def _extract_kp_from_phrases(p_list):
            post_kept = []
            for sent in p_list:
                cur_sent_kept = []
                for ph in sent:
                    if ph in cur_sent_kept:
                        continue
                    words = ph.split()
                    if len(words) >= 10:
                        continue

                    if any([w.lower() in cur_ts for w in words]):
                        cur_sent_kept.append(ph)

                post_kept.append(cur_sent_kept)
            return post_kept

        kept_np_list = _extract_kp_from_phrases(ln['np_list'])
        kept_vp_list = _extract_kp_from_phrases(ln['vp_list'])
        ret_obj = dict(
                id=ln['id'],
                np_list=kept_np_list,
                vp_list=kept_vp_list)

        kp_data.append(ret_obj)
        if cache_intermediate:
            fout.write(json.dumps(ret_obj) + '\n')


    if cache_intermediate:
        fout.close()
    return kp_data, corenlp_results



def create_planning_data(raw_data, kp_data, proc_data, domain, set_type):
    fout = open(f'{domain}/planning_{set_type}.jsonl', 'w')
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    for ln_id, (proc, kp, raw) in tqdm(enumerate(zip(proc_data, kp_data, raw_data))):
        vp_list = kp['vp_list']
        np_list = kp['np_list']
        proc_kp = _process_kp_list(np_list, vp_list)
        sents = []
        for sent in proc['proc']['sentences']:
            cur_sent = detokenize_stanford(sent['tokens'])
            sents.append(cur_sent)

        assert len(sents) == len(proc_kp)

        tgt_tokenized = []
        tgt_masked = []
        kp_tokenized = []
        kp_src_word_offset = []

        sorted_kp_sent = []
        kp_set = set()
        skipped_kp = 0

        for kp_lst, text in zip(proc_kp, sents):
            kp_order = []
            for kp in kp_lst:
                try:
                    kp_idx = text.index(kp)
                except ValueError:
                    kp_idx = fuzzy_match_kp(kp, text)

                if kp_idx is None:
                    skipped_kp += 1
                    continue
                kp_order.append((kp, kp_idx))
                kp_set.add(kp)
            kp_sorted = sorted(kp_order, key=lambda x: x[1])
            sorted_kp_sent.append([item[0] for item in kp_sorted])

            # current sentence token ids
            cur_tgt_sent_tok_ids = tokenizer.encode(text)[1:]
            cur_tgt_sent_tok = tokenizer.convert_ids_to_tokens(cur_tgt_sent_tok_ids)

            cur_kp_toks = [tokenizer.encode(kp[0])[1:-1] for kp in kp_sorted]

            cur_mask = np.zeros(len(cur_tgt_sent_tok_ids), dtype=np.int)
            removed = []
            for kp in cur_kp_toks:
                found = False
                for ix in range(0, len(cur_tgt_sent_tok_ids) - len(kp)):
                    if cur_tgt_sent_tok_ids[ix: ix + len(kp)] == kp:
                        found = True
                        cur_mask[ix: ix + len(kp)] = 1
                if not found:
                    print('{} NOT MATCHED IN \n{}'.format(tokenizer.decode(kp), text))
                    removed.append(kp)

            tgt_tokenized.extend(cur_tgt_sent_tok)
            tgt_masked = np.concatenate((tgt_masked, cur_mask))
            for ix, tok_ids in enumerate(cur_kp_toks):
                if tok_ids in removed: continue
                toks_ = tokenizer.convert_ids_to_tokens(tok_ids)
                kp_tokenized.extend(toks_)
                kp_src_word_offset.extend([ix] * len(toks_))
            kp_tokenized.append('[SEP]')
            kp_src_word_offset.append(ix + 1)

        if len(kp_set) == 0:
            print('skipping {}'.format(proc['id']))
            skipped_sample_cnt += 1
            continue

        kp_offsets, kp_tokenized, removed = calculate_kp_offsets(kp_tokenized, 
                                                                 kp_src_word_offset,
                                                                 tgt_tokenized)
        kp_set_tokenized = []
        for kp in kp_set:
            cur_tok_ids = tokenizer.encode(kp)[1:] # include [SEP]
            cur_toks = tokenizer.convert_ids_to_tokens(cur_tok_ids)
            kp_set_tokenized.extend(cur_toks)

        ret_obj = dict(
                id=proc['id'],
                tgt=tgt_tokenized,
                original_kp=proc_kp,
                kp_offsets=kp_offsets,
                kp_set=kp_set_tokenized,
                kp=kp_tokenized,
                prompt_tokens=tokenizer.tokenize(proc['prompt'])[1:-1],
                original_prompt=proc['prompt'])
        fout.write(json.dumps(ret_obj) + '\n')
    fout.close()


def create_refinement_data(raw_data, kp_data, proc_data, domain, set_type):
    """Create refinement for BART model.
    """

    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    fout = open(f'{domain}/refinement_{set_type}.jsonl', 'w')
    for ln_id, (raw, kp, proc) in enumerate(zip(raw_data, kp_data, proc_data)):
        tgt_toks = tokenizer.encode(raw['tgt'], add_special_tokens=False)
        tgt_toks = tokenizer.convert_ids_to_tokens(tgt_toks)

        kp_set = set()
        vp_list = kp['vp_list']
        np_list = kp['np_list']
        proc_kp = _process_kp_list(np_list, vp_list)
        sents = []
        for sent in proc['proc']['sentences']:
            cur_sent = utils.detokenize_stanford(sent['tokens'])
            cur_sent = cur_sent.replace('  ', ' ')
            sents.append(cur_sent)

        skipped_kp = 0
        kp_plan_str = ''

        for kp_lst, text in zip(proc_kp, sents):
            kp_order = []
            for kp in kp_lst:
                try:
                    kp_idx = text.index(kp)
                except ValueError:
                    kp_idx = fuzzy_match_kp(kp, text)

                if kp_idx is None:
                    skipped_kp += 1
                    continue
                kp_order.append((kp, kp_idx))
                kp_set.add(kp)
            kp_sorted = sorted(kp_order, key=lambda x: x[1])
            kp_plan_str += ' '.join([item[0] for item in kp_sorted]).strip() + ' <s> '

        if len(kp_set) == 0:
            continue
        kp_set_ids = [tokenizer.encode(' ' + cur_kp, add_special_tokens=False)
                          for cur_kp in kp_set]
        kp_set_toks = [tokenizer.convert_ids_to_tokens(cur_kp) for cur_kp in kp_set_ids]
        template = generate_template_for_bart(tgt_toks, kp_set_toks)

        kp_set_str = ' <s> '.join(kp_set).strip()

        ret_obj = dict(
            id=raw['id'],
            tgt=raw['tgt'],
            prompt=raw['prompt'],
            kp_plan_str=kp_plan_str.strip(),
            kp_set_str=kp_set_str,
            template=template,
        )

        fout.write(json.dumps(ret_obj) + '\n')

    fout.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["create_planning", "create_refinement"])
    parser.add_argument("--set-type", choices=['train', 'dev', 'test', 'toy'],
                        required=True)
    parser.add_argument("--domain", choices=['news', 'opinion'],
                        required=True)
    parser.add_argument("--cache-intermediate", action='store_true',
                        help="if set to True, store corenlp and chunking\
                        results to disk for faster access next time.")
    args = parser.parse_args()

    # load ids
    set_ids = load_nytimes_ids(args.domain, args.set_type)
    print(f'{len(set_ids)} ids loaded for set {args.set_type}')

    # load raw data
    raw_data = load_nytimes_rawdata(args.domain, set_ids)
    print('raw data loaded')

    kp_data, proc_data = generate_kp(raw_data,
                                         args.domain,
                                         args.set_type,
                                         args.cache_intermediate)

    if args.mode == 'create_planning':
        create_planning_data(raw_data, kp_data, proc_data,
                             args.domain, args.set_type)

    else:
        create_refinement_data(raw_data, kp_data, proc_data,
                               args.domain, args.set_type)


if __name__ == '__main__':
    main()
