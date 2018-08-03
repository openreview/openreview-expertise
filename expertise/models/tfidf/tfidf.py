from __future__ import absolute_import, print_function, unicode_literals
import sys, os

import re

from collections import defaultdict
from collections import Counter

from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

from .. import model_utils

class Model():
    def __init__(self, params=None):
        self.chunker = model_utils.extract_candidate_chunks

        self.tfidf_dictionary = corpora.Dictionary()

        self.document_tokens = []

        # a dictionary keyed on reviewer signatures, containing a BOW representation of that reviewer's Archive (Ar)
        self.bow_by_signature = defaultdict(Counter)

        # a dictionary keyed on forum IDs, containing a BOW representation of the paper (P)
        #self.bow_by_paperid = defaultdict(Counter)

        if params:
            for k, v in params.iteritems():
                print(k, v)

    def fit(self, training_data):
        """
        Fit the model to the data.

        Arguments
            @training_data: an iterator yielding training data.
            Must yield tuples in the format (<paper id>, <paper content>).

        Returns
            None

        """
        for paper_content in training_data:
            tokens = self.preprocess_content(paper_content, self.tfidf_dictionary)
            self.document_tokens += [tokens]

        # get the BOW representation for every document and put it in corpus_bows
        self.corpus_bows = [self.tfidf_dictionary.doc2bow(doc) for doc in self.document_tokens]

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

    # def score(self, signature, paperid):
    #     """
    #     Returns a score from 0.0 to 1.0, representing the degree of fit between the paper and the reviewer

    #     """
    #     paper_bow = [(id, count) for id, count in self.bow_by_paperid[paperid].iteritems()]
    #     reviewer_bow = [(id,count) for id,count in self.bow_by_signature[signature].iteritems()]
    #     forum_vector = defaultdict(lambda: 0, {idx: score for (idx, score) in self.tfidf_model[paper_bow]})
    #     reviewer_vector = defaultdict(lambda: 0, {idx: score for (idx, score) in self.tfidf_model[reviewer_bow]})

    #     return sum([forum_vector[k] * reviewer_vector[k] for k in forum_vector])

    def score(self, archive_text, paper_text):
        """
        Returns a score from 0.0 to 1.0, representing the degree of fit between the paper and the reviewer

        """
        paper_tokens = self.chunker(paper_text)
        paper_bow = [(t[0], t[1]) for t in self.tfidf_dictionary.doc2bow(paper_tokens)]

        reviewer_tokens = self.chunker(archive_text)
        reviewer_bow = [(t[0], t[1]) for t in self.tfidf_dictionary.doc2bow(reviewer_tokens)]

        forum_vector = defaultdict(lambda: 0, {idx: score for (idx, score) in self.tfidf_model[paper_bow]})
        reviewer_vector = defaultdict(lambda: 0, {idx: score for (idx, score) in self.tfidf_model[reviewer_bow]})

        return sum([forum_vector[k] * reviewer_vector[k] for k in forum_vector])


    def preprocess_content(self, content, dictionary):
        all_text = model_utils.content_to_text(content)
        tokens = self.chunker(all_text)
        dictionary.add_documents([tokens])
        return tokens

