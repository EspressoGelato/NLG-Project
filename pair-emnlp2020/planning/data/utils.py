import re
import os
import json
import time
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from nltk.chunk import regexp

STOPWORDS = [ln.strip() for ln in open('stopwords.txt')]

def chunk_tagged_sents(tagged_sent):
    grammar = r"""
        NP: {<DT|PP\$>?<JJ|JJR>*<NN.*|CD|JJ>+}
        PP: {<IN><NP>}
        VP: {<MD>?<VB.*><RP>?<NP|PP>}
        CLAUSE: {<NP><VP>}
        """

    chunker = regexp.RegexpParser(grammar, loop=2)
    chunked_sent = chunker.parse(tagged_sent)

    return chunked_sent


def get_chunks(chunked_sent, chunk_type='NP'):
    chunks = []
    raw_chunks = [subtree.leaves() for subtree in chunked_sent.subtrees()\
                    if subtree.label() == chunk_type]
    for raw_chunk in raw_chunks:
        chunk = []
        for word_tag  in raw_chunk:
            chunk.append(word_tag[0])
        chunk_str = ' '.join(chunk)
        if chunk_str not in chunks:
            chunks.append(chunk_str)
    return chunks


def detokenize_stanford(tokens):
    detokenized = ''
    for item in tokens:
        detokenized += '{}{}'.format(item['originalText'], item['after'])
    return detokenized


def _process_kp_list(np_list, vp_list):
    """Remove duplicates and merge NP/VP
    """
    output_list = []
    for np, vp in zip(np_list, vp_list):
        merged_list = np
        to_remove = []
        for item in vp:
            if item in merged_list:
                continue

            processed = False
            for exist in merged_list:
                # if cur is substring of existing, remove existing if
                # the extra part is stopword, otherwise pass cur
                if item in exist:
                    extra = exist.replace(item, '').strip().split()
                    contains_nonstop = any([w.lower() not in STOPWORDS for w in extra])
                    if not contains_nonstop:
                        # remove exist
                        to_remove.append(exist)
                        merged_list.append(item)

                    processed = True
                    break

                # if existing is substring of cur, remove existing if
                # the extra part is non-stopword, otherwise pass cur
                if exist in item:
                    extra = item.replace(exist, '').strip().split()
                    contains_nonstop = any([w.lower() not in STOPWORDS for w in extra])
                    if contains_nonstop:
                        to_remove.append(exist)
                        merged_list.append(item)

                    processed = True
                    break

            if processed:
                continue

            merged_list.append(item)
        output_list.append([item for item in merged_list if not item in to_remove])
    return output_list

def fuzzy_match_kp(kp, text):
    """Match kp with text if no exact match was found.

    Args:
        kp (str):  such as `2.5 B worth`
        text (str): such as `$2.5B worth of oil`

    We consider shrink spaces between tokens in kp.
    """

    # 1-space
    for ch_ix, ch in enumerate(kp):
        if ch != ' ': continue
        new_kp = kp[:ch_ix] + kp[ch_ix + 1:]
        if new_kp in text:
            return text.index(new_kp)

    for ch_ix, ch in enumerate(kp):
        if ch != ' ': continue
        new_kp = kp[:ch_ix] + kp[ch_ix + 1:]
        secondary_match = fuzzy_match_kp(new_kp, text)
        if secondary_match is not None:
            return secondary_match
    return None


def calculate_kp_offsets(kp_words, kp_ids, tgt_words):
    """For a given sequence of tokenized keyphrases, calculate sentence word
    offset for each tokens.

    Args:
        kp_words: a list of Bert subwords
        kp_src_word_offset: a list of ids denoting the KP id
        tgt_words: a list of target subwords
    """
    assert len(kp_words) == len(kp_ids)

    sent_kp = []
    cur_chunk = []
    for w, ix in zip(kp_words, kp_ids):
        if w == '[SEP]':
            cur_chunk.append((w, ix))
            sent_kp.append(cur_chunk)
            cur_chunk = []
        else:
            cur_chunk.append((w, ix))

    sent_tgt = []
    cur_chunk = []
    for w in tgt_words:
        if w == '[SEP]':
            cur_chunk.append(w)
            sent_tgt.append(cur_chunk)
            cur_chunk = []
        else:
            cur_chunk.append(w)


    assert len(sent_tgt) == len(sent_kp)
    kp_offsets = []
    new_kp = []
    removed = 0
    last_ix = -1

    for kp_s, tgt_s in zip(sent_kp, sent_tgt):
        last_kp_ix = 0
        for wix, (w, ix) in enumerate(kp_s):
            if last_ix == ix:
                kp_offsets.append(kp_offsets[-1] + 1)
            elif tgt_s.count(w) == 1:
                kp_offsets.append(tgt_s.index(w))
            elif tgt_s.count(w) > 1:
                # if multiple occurence, pick the one with similar context,
                # if none found then use the earliest occurrence
                ix = _match_best(kp_s, wix, tgt_s)
                kp_offsets.append(ix)
            else:
                # print('can\'t find keyword {} in sent {}'.format(w, tgt_s))
                removed += 1
                continue
                # kp_offsets.append(-1)
                # raise ValueError('can\'t find keyword {} in sent {}'.format(w, tgt_s))
            new_kp.append(w)
    assert len(kp_offsets) == len(new_kp)
    return kp_offsets, new_kp, removed


def get_nytimes_topic_signatures(domain):
    path = f"{domain}/loglikehood_ratio.jsonl"
    id2ts = dict()
    t0 = time.time()

    for ln in open(path):
        cur_obj = json.loads(ln)
        cur_lst = []
        for word in cur_obj['ratio_ranked_words']:
            if word[1] < 10.83:
                break
            cur_lst.append(word[0])
        id2ts[cur_obj['id']] = cur_lst
    print('{} topic signature loaded in {:.2f} secs'.format(domain, time.time() - t0))
    return id2ts

def get_facts_topic_signatures():
    path = f"fact_loglikelihood_ratio.jsonl"
    id2ts = dict()
    t0 = time.time()

    for ln in open(path):
        cur_obj = json.loads(ln)
        cur_lst = []
        for word in cur_obj['ratio_ranked_words']:
            if word[1] < 10.83:
                break
            cur_lst.append(word[0])
        id2ts[cur_obj['id']] = cur_lst
    print('topic signature loaded in {:.2f} secs'.format(time.time() - t0))
    return id2ts


def extract_body(full_article, domain):
    """Extract body and remove boilerplates.

    For both domains (news, opinion), remove LEAD (0-th element in full_article).
    Change '' into "
    For opinion:
        1. remove 1-st line if it's ['to the editor', '']
        2. remove name, location, date line (usually the last element)

    For news:
        ...
    """

    TO_THE_EDITOR = 'to the editor:'

    result = ''
    skipped_content = []
    end = False
    for ln_id, ln in enumerate(full_article):
        if end:
            skipped_content.extend(full_article[ln_id:])
            break

        if ln_id == 0:
            if 'LEAD' in ln:
                skipped_content.append('LEAD:' + ln)
                continue # skip LEAD

        if ln_id <= 1:
            if TO_THE_EDITOR in ln.lower():
                ln = ln[len(TO_THE_EDITOR):].strip()
                if len(ln) > 0:
                    result += ln + ' '
                continue

        ln = ln.strip()
        ln = ln.replace("''", '"')

        if domain == 'opinion' and ln_id >= len(full_article) - 3 and ln_id < len(full_article) - 1:
            toks = ln.split()
            # too short
            if len(toks) <= 2:
                skipped_content.append(ln)
                continue


            # (almost) all capitals
            if len([w for w in toks if w[0].lower() != w[0]]) / len(toks) > 0.5:
                end = True
                skipped_content.append('ALMOST-ALL-CAPITAL:' + ln)
                continue

        if domain == 'opinion' and ln_id == len(full_article) - 1:
            sents = sent_tokenize(ln)
            for sent in sents:
                all_capitals = re.findall(r'\b([A-Z]+)\b', sent)
                all_capitals = [w for w in all_capitals if w != 'I']
                sent_total_toks = sent.split()
                if len(all_capitals) >= len(sent_total_toks) - 2:
                    skipped_content.append('ALL-CAPITAL-EXIST({} - {}):'.format(len(sent_total_toks), ','.join(all_capitals)) + sent)
                else:
                    toks = sent.split()
                    if len([w for w in toks if w[0].lower() != w[0]]) / len(toks) > 0.5:
                        skipped_content.append('ALMOST-ALL-CAPITAL(2):' + sent)
                    else:
                        result += sent + ' '
            continue

        result += ln + ' '

    return result.strip(), skipped_content

def _match_best(kp_s, word_idx, tgt_s):
    """Find the best match word for a word in a given keyphrase. First check
    if context can uniquely identify the word, if not return the one that
    appears first.
    """

    found_ix = -1
    #to_check_word = kp_s[word_idx]
    to_check_word = kp_s[word_idx][0]
    for tgt_wix, tgt_w in enumerate(tgt_s):
        if tgt_w != to_check_word:
            continue

        left_matched, right_matched = False, False
        # only use words of the same KP chunk as context
        cur_kp_ix = kp_s[word_idx][1]
        if word_idx >= 1:
            left_kp_ix = kp_s[word_idx - 1][1]
            if left_kp_ix == cur_kp_ix:
                left_context = kp_s[word_idx - 1][0]
                left_tgt = tgt_s[tgt_wix - 1] if tgt_wix >= 1 else None
                if left_context and left_tgt and left_context == left_tgt:
                    left_matched = True

        if word_idx < len(kp_s) - 1:
            right_kp_ix = kp_s[word_idx + 1][1]
            if right_kp_ix == cur_kp_ix:
                right_context = kp_s[word_idx + 1][0]
                right_tgt = tgt_s[tgt_wix + 1] if tgt_wix < len(tgt_s) else None
                if right_context and right_tgt and right_context == right_tgt:
                    right_matched = True

        '''
        left_context = kp_s[word_idx - 1] if word_idx >= 1 else None
        left_tgt = tgt_s[tgt_wix - 1] if tgt_wix >= 1 else None
        if left_context and left_tgt and left_context == left_tgt:
            left_matched = True

        right_context = kp_s[word_idx + 1] if word_idx < len(kp_s) - 1 else None
        right_tgt = tgt_s[tgt_wix + 1] if tgt_wix < len(tgt_s) else None
        if right_context and right_tgt and right_context == right_tgt:
            right_matched = True
        '''

        if left_matched or right_matched:
            found_ix = tgt_wix

    if found_ix == -1:
        found_ix = tgt_s.index(kp_s[word_idx][0])
    return found_ix


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


