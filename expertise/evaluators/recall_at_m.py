from __future__ import print_function, absolute_import

import sys, os

from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')


class Evaluator():
    """
    An Evaluator instance that evaluates
    recall_at_m =
        (number of papers reviewer bid positively on in top M) /
        (total number of papers reviewer bid positively on)

    This evaluation method requires us to look at the bids, so we import them
    from somewhere in the __init__() method.

    """

    def __init__(self, bids_by_forum, m=50):
        # datapath = os.path.join(os.path.dirname(__file__), '../samples/uai_data')
        # self.data = utils.load_obj(datapath)
        # self.bids_by_forum = self.data['bids_by_forum']
        self.bids_by_forum = bids_by_forum
        self.m_values = range(m)

    def recall_at_m(self, ranked_signatures, forum, m):
        '''
        Among the top M users in the model's rank list,
        what percentage of all high bidders are included?

        '''

        positive_bids = self.get_pos_bids_for_forum(forum)
        positive_signatures = [bid['signature'] for bid in positive_bids]

        if not positive_signatures:
            # what value should be returned if there are no positives?
            return 1.0
        else:
            all_bids = self.get_all_bids_for_forum(forum)
            bid_signatures = [bid['signature'] for bid in all_bids]

            # only rank the reviewers who made a bid of some kind
            # filtered_ranked_signatures = [r for r in ranked_signatures if r in bid_signatures]
            # topM = filtered_ranked_signatures[:m]
            topM = ranked_signatures[:m]

            positive_signatures_from_topM = [r for r in positive_signatures if r in topM]

            return float(len(positive_signatures_from_topM))/float(len(positive_signatures))

    def evaluate(self, ranklists):
        """
        Evaluate the model using a ranked list. Either you can evaluate using a single ranked list or
        evaluate against each individual query and average their precision scores

        Arguments
            @ranklists: A list of tuples.
            The 0th index of the tuple contains the forum ID of the rank list being evaluated.
            The 1st index of the tuple contains a list of reviewer IDs, in order of expertise score.

        Returns
            a generator object that yields an array of scores for each ranked list. If only one
            score is needed, return the score in an array by itself.
        """

        for forum, rank_list in ranklists:
            ranked_signatures = [rank.split(";")[0] for rank in rank_list]
            scores = []
            for m in self.m_values:
                scores.append(self.recall_at_m(ranked_signatures, forum, m))
            yield forum, scores

    def get_all_bids_for_forum(self, forum_id):
        """ Returns all bids for the forum_id """
        forum_bids = self.bids_by_forum[forum_id]
        return [{"signature": bid.signatures[0], "bid": bid.tag} for bid in forum_bids]

    def get_pos_bids_for_forum(self, forum_id):
        """ Get all of the positive bids for a forum """
        positive_labels = ["I want to review", "I can review"]
        forum_bids = self.get_all_bids_for_forum(forum_id)
        return [bid for bid in forum_bids if bid["bid"] in positive_labels]
