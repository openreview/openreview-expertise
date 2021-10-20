from __future__ import absolute_import, print_function, unicode_literals

from collections import defaultdict

from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity


class Model:
    def __init__(self, kp_archives_by_paperid, kp_archives_by_userid):

        self.dictionary = corpora.Dictionary()

        # self.bow_by_userid = defaultdict(Counter)
        # self.bow_by_paperid = defaultdict(Counter)

        self.all_documents = []

        self.kp_archives_by_paperid = kp_archives_by_paperid
        self.kp_archives_by_userid = kp_archives_by_userid

        for archive in self.kp_archives_by_paperid.values():
            for token_list in archive:
                self.dictionary.add_documents([token_list])
                self.all_documents += [token_list]

        for archive in self.kp_archives_by_userid.values():
            for token_list in archive:
                self.dictionary.add_documents([token_list])
                self.all_documents += [token_list]

        self.corpus_bows = [self.dictionary.doc2bow(doc) for doc in self.all_documents]
        self.tfidf = TfidfModel(self.corpus_bows)

    def fit(self):
        """
        Fit the TFIDF model

        each argument should be a list of lists, where each inner list is a list of keyphrases.

        e.g.

        submission_kps = [
            ['deep_learning', 'natural_language_processing'],
            ['neural_imaging', 'fmri', 'functional_magnetic_resonance']
        ]
        """

        self.bow_archives_by_paperid = {
            userid: [self.dictionary.doc2bow(doc) for doc in archive]
            for userid, archive in self.kp_archives_by_paperid.items()
        }

        self.bow_archives_by_userid = {
            userid: [self.dictionary.doc2bow(doc) for doc in archive]
            for userid, archive in self.kp_archives_by_userid.items()
        }

        flattened_archives = [
            bow for archive in self.bow_archives_by_paperid.values() for bow in archive
        ]

        self.index = SparseMatrixSimilarity(
            [self.tfidf[bow] for bow in flattened_archives],
            num_features=len(self.dictionary),
        )

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

        scores = [
            (signature, self.score(signature, note_record))
            for signature, _ in self.bow_by_userid.iteritems()
        ]
        rank_list = [
            signature
            for signature, score in sorted(scores, key=lambda x: x[1], reverse=True)
        ]

        return rank_list

    def score(self, reviewer_tokens, paper_tokens):
        """
        Returns a score from 0.0 to 1.0, representing the degree of fit between the paper and the reviewer

        """

        paper_bow = [(t[0], t[1]) for t in self.dictionary.doc2bow(paper_tokens)]
        reviewer_bow = [(t[0], t[1]) for t in self.dictionary.doc2bow(reviewer_tokens)]

        forum_vector = defaultdict(
            lambda: 0, {idx: score for (idx, score) in self.tfidf[paper_bow]}
        )
        reviewer_vector = defaultdict(
            lambda: 0, {idx: score for (idx, score) in self.tfidf[reviewer_bow]}
        )

        return sum([forum_vector[k] * reviewer_vector[k] for k in forum_vector])
