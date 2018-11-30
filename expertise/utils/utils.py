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

def jsonl_reader(jsonl_file):
    '''
    Utility function for lazily reading a .jsonl file.
    '''
    with open(jsonl_file) as f:
        for line in f.readlines():
            yield json.loads(line.replace('\n',''))

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

def content_to_text(content, fields=['title', 'abstract', 'pdftext']):
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

