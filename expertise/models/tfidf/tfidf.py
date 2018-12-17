from __future__ import absolute_import, print_function, unicode_literals
import sys, os

import re

from collections import defaultdict
from collections import Counter

from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

from expertise.preprocessors import pos_regex


class Model():
    def __init__(self):

        self.tfidf_dictionary = corpora.Dictionary()

        self.document_tokens = []

        # a dictionary keyed on reviewer signatures, containing a BOW representation of that reviewer's Archive (Ar)
        self.bow_by_signature = defaultdict(Counter)

        # a dictionary keyed on forum IDs, containing a BOW representation of the paper (P)
        #self.bow_by_paperid = defaultdict(Counter)

        self.preprocess_content = pos_regex.extract_candidate_words

    def fit(self, keyphrases):
        """
        Fit the TFIDF model

        keyphrases should be a list of lists, where each inner list is a list of keyphrases.

        e.g.

        keyphrases = [
            ['deep_learning', 'natural_language_processing'],
            ['neural_imaging', 'fmri', 'functional_magnetic_resonance']
        ]
        """
        for tokens in keyphrases:
            self.tfidf_dictionary.add_documents([tokens])
            self.document_tokens += [tokens]

        # get the BOW representation for every document and put it in corpus_bows
        self.corpus_bows = [self.tfidf_dictionary.doc2bow(doc) for doc in self.document_tokens]

        # print(self.corpus_bows)
        # generate a TF-IDF model based on the entire corpus's BOW representations
        self.tfidf_model = TfidfModel(self.corpus_bows)

    def predict(self, note_record):
        """
        predict() should return a list of openreview user IDs, in descending order by
        expertise score in relation to the test record.

        Arguments
            @test_record: a note record (dict) representing the note to rank against.

            Testing records should have a "forum" field. This means that the record
            is identified in OpenReview by the ID listed in that field.

        Returns
            a list of reviewer IDs in descending order of expertise score

        """

        scores = [(signature, self.score(signature, note_record)) for signature, _ in self.bow_by_signature.iteritems()]
        rank_list = [signature for signature, score in sorted(scores, key=lambda x: x[1], reverse=True)]

        return rank_list

    def score(self, archive_content, paper_content):
        """
        Returns a score from 0.0 to 1.0, representing the degree of fit between the paper and the reviewer

        """
        paper_tokens = self.preprocess_content(paper_content)
        paper_bow = [(t[0], t[1]) for t in self.tfidf_dictionary.doc2bow(paper_tokens)]

        reviewer_tokens = self.preprocess_content(archive_content)
        reviewer_bow = [(t[0], t[1]) for t in self.tfidf_dictionary.doc2bow(reviewer_tokens)]

        forum_vector = defaultdict(lambda: 0, {idx: score for (idx, score) in self.tfidf_model[paper_bow]})
        reviewer_vector = defaultdict(lambda: 0, {idx: score for (idx, score) in self.tfidf_model[reviewer_bow]})

        return sum([forum_vector[k] * reviewer_vector[k] for k in forum_vector])


