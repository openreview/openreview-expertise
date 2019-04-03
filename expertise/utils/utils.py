from __future__ import print_function, absolute_import, unicode_literals
import codecs
import sys
import itertools
import string
import nltk
import torch
import json
import nltk
import re
import pickle
import csv
from collections import defaultdict
import math, random

def partition(list_, partition_id=0, num_partitions=1):
    '''
    Given a list, partitions the list according to `num_partitions`.
    This function is useful for parallelization.

    Example:
    >>> for i in partition(range(100), partition_id=3, num_partitions=20):
    >>>     print(i)
    3
    23
    43
    63
    83

    '''
    for item_idx, item in enumerate(list_):
        if item_idx % num_partitions == partition_id:
            yield item
        else:
            pass

def dump_csv(filepath, data):
    '''
    Writes .csv files in a specific format preferred by some IESL students:
    tab-delimited columns, with keyphrases separated by spaces.
    '''
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for line in data:
            writer.writerow(line)

def dump_pkl(filepath, data):
    '''
    dump an object to a pickle file
    '''
    with open(filepath, 'wb') as f:
        f.write(pickle.dumps(data))

def load_pkl(filepath):
    '''
    load an object from a pickle file
    '''
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def dump_jsonl(filepath, data):
    with open(filepath, 'w') as f:
        for data_dict in data:
            f.write(json.dumps(data_dict) + '\n')

def dump_json(filepath, data_dict):
    with open(filepath, 'w') as f:
        json.dump(data_dict, f)

def jsonl_reader(jsonl_file):
    '''
    Utility function for lazily reading a .jsonl file.
    '''
    with open(jsonl_file) as f:
        for line in f:
            yield json.loads(line.rstrip())

def holdouts(full_list):
    '''
    Given a list of items, returns a list of tuples.

    Each tuple contains two items:
        the "holdout": one of the items in the full list
        the "remainder": the rest of the items in the full list

    This function returns a list of these tuples. This list should
        have the same length as the input.

    >>> full_list = ['a','b','c']
    >>> holdouts(full_list)
    [
        ('a', ['b', 'c']),
        ('b', ['a', 'c']),
        ('c', ['a', 'b'])
    ]

    '''
    holdouts_list = []
    for index, item in enumerate(full_list):
        holdout = item
        remainder = full_list[:index] + full_list[index+1:]
        holdouts_list.append((holdout, remainder))
    return holdouts_list

def load_labels(filename):
    '''
    '''
    labels_by_forum = defaultdict(dict)
    scores_by_forum = defaultdict(dict)

    for data in jsonl_reader(filename):
        forum = data['source_id']
        reviewer = data['target_id']
        label = data['label']
        score = data['score']
        labels_by_forum[forum][reviewer] = int(label)
        scores_by_forum[forum][reviewer] = float(score)


    result_labels = []
    result_scores = []

    for forum, labels_by_reviewer in labels_by_forum.items():
        scores_by_reviewer = scores_by_forum[forum]

        reviewer_scores = list(scores_by_reviewer.items())
        reviewer_labels = list(labels_by_reviewer.items())

        sorted_labels = [label for _, label in sorted(reviewer_labels)]
        sorted_scores = [score for _, score in sorted(reviewer_scores)]

        result_labels.append(sorted_labels)
        result_scores.append(sorted_scores)

    return result_labels, result_scores

def get_bids_by_forum(dataset):
    binned_bids = {val: [] for val in dataset.bid_values}

    positive_labels = dataset.positive_bid_values

    users_w_bids = set()
    for bid in dataset.bids():
        binned_bids[bid.tag].append(bid)
        users_w_bids.update(bid.signatures)

    bids_by_forum = defaultdict(list)
    for bid in dataset.bids():
        bids_by_forum[bid.forum].append(bid)

    pos_and_neg_signatures_by_forum = {}

    # Get pos bids for forum
    for forum_id, forum_bids in bids_by_forum.items():
        forum_bids_flat = [{"signature": bid.signatures[0], "bid": bid.tag} for bid in forum_bids]
        neg_bids = [bid for bid in forum_bids_flat if bid["bid"] not in positive_labels]
        neg_signatures = [bid['signature'] for bid in neg_bids]
        pos_bids = [bid for bid in forum_bids_flat if bid["bid"] in positive_labels]
        pos_signatures = [bid['signature'] for bid in pos_bids]
        pos_and_neg_signatures_by_forum[forum_id] = {}
        pos_and_neg_signatures_by_forum[forum_id]['positive'] = pos_signatures
        pos_and_neg_signatures_by_forum[forum_id]['negative'] = neg_signatures

    return pos_and_neg_signatures_by_forum

def format_bid_labels(eval_set_ids, bids_by_forum):
    for forum_id in eval_set_ids:
        for reviewer in bids_by_forum[forum_id]['positive']:
            yield {'source_id': forum_id, 'target_id': reviewer, 'label': 1}

        for reviewer in bids_by_forum[forum_id]['negative']:
            yield {'source_id': forum_id, 'target_id': reviewer, 'label': 0}

def split_ids(ids):
    random.seed(a=3577057385653016827)

    forums = sorted(ids)
    random.shuffle(forums)

    idxs = (math.floor(0.6 * len(forums)), math.floor(0.7 * len(forums)))

    train_set_ids = forums[:idxs[0]]
    dev_set_ids = forums[idxs[0]:idxs[1]]
    test_set_ids = forums[idxs[1]:]

    return train_set_ids, dev_set_ids, test_set_ids

def format_data_bids(train_set_ids, bids_by_forum, reviewer_kps, submission_kps, max_num_keyphrases=None):
    '''
    Formats bid data into source/positive/negative triplets.
    (This function is written specifically to handle keyphrase-based data.)

    "source" represents the paper being compared against.
    "positive" represents a paper authored by a reviewer who bid highly on
        the source paper.
    "negative" represents a paper authored by a reviewer who bid lowly on
        the source paper.

    '''

    for forum_id in train_set_ids:
        if forum_id in submission_kps:

            forum_kps = [kp for kp in submission_kps[forum_id]][:max_num_keyphrases]
            forum_pos_signatures = sorted(bids_by_forum[forum_id]['positive'])
            forum_neg_signatures = sorted(bids_by_forum[forum_id]['negative'])

            pos_neg_pairs = itertools.product(forum_pos_signatures, forum_neg_signatures)
            for pos, neg in pos_neg_pairs:
                if pos in reviewer_kps and neg in reviewer_kps:
                    data = {
                        'source': forum_kps[:max_num_keyphrases],
                        'source_id': forum_id,
                        'positive': reviewer_kps[pos][:max_num_keyphrases],
                        'positive_id': pos,
                        'negative': reviewer_kps[neg][:max_num_keyphrases],
                        'negative_id': neg
                    }

                    yield data

def format_data_heldout_authors(kp_lists_by_reviewer, kps_by_reviewer):
    '''
    Formats reviewer data into source/positive/negative triplets.
    (This function is written specifically to handle keyphrase-based data.)

    "source" represents the paper being compared against.
    "positive" represents another paper written by the author that wrote
        the source paper.
    "negative" represents a paper written by an author that did not write
        the source paper.
    '''

    for source_reviewer, reviewer_kp_lists in kp_lists_by_reviewer.items():
        print('processing source reviewer',source_reviewer)
        '''
        kp_lists_by_reviewer is a dict, keyed on reviewer ID, where each value is a list of lists.
            each outer list corresponds to that reviewer's papers.
            each inner list contains the keyphrases for that paper.

        kps_by_reviewer is a dict, also keyed on reviewer_id, where each value is a list of all
            keyphrases for the reviewer.
        '''

        negative_reviewers = [n for n in kps_by_reviewer if n != source_reviewer]

        for source_kps, remainder_kp_lists in holdouts(reviewer_kp_lists):
            '''
            source_kps is a list of keyphrases representing one of the source_reviewer's papers.
            remainder_kp_lists is a list of lists representing the other papers.
            '''

            # positive_kps = [kp for kp_list in remainder_kp_lists for kp in kp_list]
            for positive_kps in remainder_kp_lists:

                # pick a random reviewer (who is not the same as the source/positive reviewer)
                negative_reviewer = random.sample(negative_reviewers, 1)[0]
                negative_kps = kps_by_reviewer[negative_reviewer]

                data = {
                    'source': source_kps,
                    'source_id': source_reviewer,
                    'positive': positive_kps,
                    'positive_id': source_reviewer,
                    'negative': negative_kps,
                    'negative_id': negative_reviewer
                }

                yield data


def strip_nonalpha(text):
    '''
    lowercases words
    removes characters outside of the 26-letter english alphabet
    '''
    new_text = []
    lowercase_alphabet = [char for char in 'abcdefghijklmnopqrstuvwxyz']

    for word in text.split(' '):
        new_word = []

        for char in word.lower():
            if char in lowercase_alphabet:
                new_word.append(char)

        new_text.append(''.join(new_word))

    return ' '.join(new_text)


def read_scores(file):
    '''
    Reads a file with scores for paper-reviewer pairs
    '''
    score_matrix = {}
    with open(file) as f:
        lines = [line.replace('\n','') for line in f.readlines()]

    for line in lines:
        note_id, reviewer_id, score = eval(line)
        if note_id not in score_matrix:
            score_matrix[note_id] = {}
        if reviewer_id not in score_matrix[note_id]:
            score_matrix[note_id][reviewer_id] = float(score)
        else:
            raise('pair already seen ', note_id, reviewer_id, score)

    return score_matrix

def matrix_to_ranklists(score_matrix):
    '''
    Converts score_matrix, a dict keyed on [paper_id] and [reviewer_id], into
    a list of ranklists.

    inputs @score_matrix:
    score_matrix['abcXYZ']['~Michael_Spector'] = score of affinity between
    "~Michael_Spector1" and paper with forum "abcXYZ")

    returns @ranklists:
    [
        'abcXYZ': ['~Michael_Spector1', '~Melisa_Bok1', ... '~Pamela_Mandler1']
        '123ABC': ['~Pamela_Mandler1', '~Michael_Spector1', ... '~Melisa_Bok1']
        ...
    ]

    Each ranklist in @ranklists is a list of reviewer IDs in order of score,
    highest to lowest.


    '''
    ranklists = []
    for paper_id in score_matrix:
        paper_scores = score_matrix[paper_id]
        ranklist_w_scores = sorted(
            [(reviewer_id, paper_scores[reviewer_id]) for reviewer_id in paper_scores],
            key=lambda x: x[1],
            reverse=True
        )
        ranklist = ['{};{}'.format(reviewer_score_tuple[0], reviewer_score_tuple[1]) for reviewer_score_tuple in ranklist_w_scores]
        ranklist_tuple = (paper_id, ranklist)
        ranklists.append(ranklist_tuple)

    return ranklists

def content_to_text(content, fields=['title', 'abstract', 'fulltext']):
    '''
    Given a dict "content", such as the content field in an OpenReview record,
    return a string that is the concatenation of the fields in "fields".

    Example:

    >>> sample_note_content = {
    ...     'title': 'My Title',
    ...     'abstract': 'This is the abstract of my paper.'
    ... }
    >>> content_to_text(sample_note_content)
    'My Title This is the abstract of my paper.'

    '''
    return ' '.join([content.get(field, '') for field in fields])

def preprocess(text, mode='chunks', stemmer=None):
    if mode=='chunks':
        return extract_candidate_chunks(text, stemmer=stemmer)
    if mode=='words':
        return extract_candidate_words(text, stemmer=stemmer)

def extract_candidate_chunks(text, grammar=r'NP: {<JJ>*<NN>}', delimiter='_', stemmer=None):

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    all_chunks = list(itertools.chain.from_iterable(
        nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents)
    )

    # join constituent chunk words into a single chunked phrase

    if stemmer:
        stem = stemmer.stem
    else:
        stem = lambda x: x

    candidates = []
    for key, group in itertools.groupby(all_chunks, lambda word_pos_chunk_triple: word_pos_chunk_triple[2] != 'O'):
        if key:
            words = []
            for word, pos, chunk in group:
                try:
                    word = stem(word)
                except IndexError:
                    print("word unstemmable:", word)
                words.append(word)
            candidates.append(delimiter.join(words).lower())

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS']), stemmer=None):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words

    if stemmer!=None:
        candidates = [stemmer.stem(word.lower()) for word, tag in tagged_words
                      if tag in good_tags and word.lower() not in stop_words
                      and not all(char in punct for char in word)]
    else:
        candidates = [word.lower() for word, tag in tagged_words
                      if tag in good_tags and word.lower() not in stop_words
                      and not all(char in punct for char in word)]

    return candidates



'''
Below are utils from Justin's code
'''

def file_lines(filename):
    with open(filename) as f:
        for line in f.readlines():
            yield line.decode(codec)

def row_wise_dot(tensor1, tensor2):
    return torch.sum(tensor1 * tensor2, dim=1, keepdim=True)

def __filter_json(the_dict):
    res = {}
    for k in the_dict.keys():
        if type(the_dict[k]) is str or type(the_dict[k]) is float or type(the_dict[k]) is int or type(the_dict[k]) is list:
            res[k] = the_dict[k]
        elif type(the_dict[k]) is dict:
            res[k] = __filter_json(the_dict[k])

    return res

def save_dict_to_json(the_dict,the_file):
    with open(the_file, 'w') as fout:
        fout.write(json.dumps(__filter_json(the_dict)))
        fout.write("\n")

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
             'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
             'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
             'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
             'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
             'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
             't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
             "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
             "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def ft_embed(ft_model, kps_list):
    D = ft_model.get_dimension()
    embeds = np.zeros((len(kps_list), D))
    for idx, kp in enumerate(kps_list):
        words = [w.lower() for w in kp.split("_") if w not in stopwords]
        emb_sum = np.sum(np.array([ft_model.get_word_vector(w) for w in words]), axis=0)
        emb_avg = emb_sum / len(words)
        embeds[idx, :] = emb_avg
    return embeds

