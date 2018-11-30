from __future__ import print_function
import os
import json
import itertools
import string
import nltk

from .. import utils

def keyphrases(text):
    '''
    keyphrases should accept one argument, text, which is a string.

    returns a list of normalized keyphrases.
    '''
    return extract_candidate_chunks(text)

# Deprecated
def read_keyphrases(data_dir):
    '''
    Given a directory containing reviewer archives or submissions,
    generate a dict keyed on signatures whose values are sets of keyphrases.

    The input directory should contain .jsonl files. Files representing
    reviewer archives should be [...] TODO: Finish this.
    '''
    print('function deprecated')
    # for filename in os.listdir(data_dir):
    #     filepath = os.path.join(data_dir, filename)

    #     file_id = filename.replace('.jsonl', '')
    #     print(file_id)

    #     keyphrases = []

    #     with open(filepath) as f:
    #         for line in f.readlines():
    #             if line.endswith('\n'):
    #                 line = line[:-1]

    #             record = json.loads(line)
    #             content = record['content']

    #             record_text_unfiltered = utils.content_to_text(content, fields=['title', 'abstract', 'fulltext'])
    #             record_text_filtered = utils.strip_nonalpha(record_text_unfiltered)

    #             keyphrases.extend(extract_candidate_chunks(record_text_filtered))

    #     yield file_id, keyphrases

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
