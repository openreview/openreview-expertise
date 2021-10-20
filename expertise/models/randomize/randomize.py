import numpy as np


class Model:
    def __init__(self):
        pass

    def fit(self, **kwargs):
        pass

    def score(self):
        return np.random.random()

    # def __init__(self):
    #     self.reviewers = set()
    #     pass

    # def fit(self, train_data, archive_data):
    #     self.reviewers = set([record['reviewer_id'] for record in archive_data])

    # def predict(self, note_record):
    #     scores = [(signature, self.score(signature, note_record['forum'])) for signature in self.reviewers]
    #     return [signature for signature, score in sorted(scores, key=lambda x: x[1], reverse=True)]

    # def score(self, signature, note_record):
    #     return np.random.random()
