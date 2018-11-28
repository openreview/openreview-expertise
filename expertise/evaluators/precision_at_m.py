from operator import itemgetter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from openreview_matcher.evals import base_evaluator
from openreview_matcher import utils

matplotlib.style.use('ggplot')


class Evaluator(base_evaluator.Evaluator):
    """
    An Evaluator instance that evaluates
    precision_at_m =
        (number of papers reviewers bid positively on in top M) /
        (total number of papers retrieved)

    This evaluation method requires us to look at the bids, so we import 
    them from somewhere in the __init__() method
    """

    def __init__(self, params=None):
        datapath = params["data_path"]
        self.m_values = params["m_values"]
        self.data = utils.load_obj(datapath)
        self.bids_by_forum = self.data["bids_by_forum"]

    def evaluate(self, ranklists):
        """
        Evaluate the model using a ranked list. Either you can evaluate using a single ranked list or 
        evaluate against each individual query and average their precision scores
        
        Arguments
            @ranklists: a list of tuples.
            The 0th index of the tuple contains the forum ID of the rank of the list being evaluated
            The 1st index of the tuple contains a list of reviewer IDS, in order of expertise score

        Returns
            a generator object that yields an array of scores for each ranked list. If only one score
            is need, return the score in an array by itself

        """

        return self.evaluate_using_single_rank(ranklists)

    def evaluate_using_individual_queries(self, ranklists):
        """ Evaluate using individual query ranks """

        for forum, rank_list in ranklists:
            scores = []
            for m in self.m_values:
                positive_labels = ["I want to review", "I can review"]
                positive_bids = [bid.signatures[0].encode('utf-8') for bid in self.bids_by_forum[forum] if bid.tag in positive_labels]
                relevant_reviewers = [1 if reviewer_id in positive_bids else 0 for reviewer_id in rank_list]
                precision = self.precision_at_m(relevant_reviewers, m)
                scores.append(precision)
            yield forum, scores

    def setup_ranked_list(self, rank_list):
        """
        Setup the single ranked list for a model 
        Combines all of the individual query ranks into one single rank 
        """
        new_rank_list = []

        for forum, rank_list in rank_list:
            for reviewer_score in rank_list:
                reviewer = reviewer_score.split(";")[0]
                score = float(reviewer_score.split(";")[1])
                has_bid = self.reviewer_has_bid(reviewer, forum)  # filter for reviewers that gave a bid value
                if has_bid:
                    new_rank_list.append((reviewer, score, forum))
        ranked_reviewers = sorted(new_rank_list, key=itemgetter(1), reverse=True)
        return ranked_reviewers

    def reviewer_has_bid(self, reviewer, paper):
        """ Returns True if the reviewer bid on that 'paper' """
        paper_bids = self.bids_by_forum[paper]
        has_bid = [True if bid.signatures[0] == reviewer.decode("utf-8") else False for bid in paper_bids][0]
        return has_bid

    def get_bid_for_reviewer_paper(self, reviewer, paper):
        """ 
        Gets the bid for the reviewer and the paper 
        Returns 0 if the bid is not relevant and 1 if the bid is relevant
        """
        positive_labels = ['I want to review', 'I can review']
        paper_bids = self.bids_by_forum[paper]
        bid_value = [1 if bid.tag in positive_labels else 0 for bid in paper_bids if
                     bid.signatures[0] == reviewer.decode('utf-8')]
        if len(bid_value) > 0:
            return bid_value[0]
        else:
            return 0

    def evaluate_using_single_rank(self, rank_list):
        """
        Evaluate against a single ranked list computed by the model  
        """

        ranked_reviewers = self.setup_ranked_list(rank_list)

        scores = []

        positive_bids = 0
        for reviewer, score, forum in ranked_reviewers:
            bid = self.get_bid_for_reviewer_paper(reviewer, forum)
            if bid == 1:
                positive_bids +=1

        for m in range(1, len(ranked_reviewers) + 1):
            topM = ranked_reviewers[0: m]
            topM = map(lambda reviewer: (reviewer[0], self.get_bid_for_reviewer_paper(reviewer[0], reviewer[2])), topM)
            pos_bids_from_topM = [bid for bid in topM if bid[1] == 1]
            precision = float(len(pos_bids_from_topM)) / float(m)  # precision => relevant bids retrieved / # of retrieved
            scores.append((m, precision))

        return scores
    def precision_at_m(self, ranked_list, m):
        """ 
        Computes precision at M 
        
        Arguments:
            ranked_list: ranked list of reviewers for a forum where each entry is either a 0 or 1
                        1 -  reviewer that reviewer wanted to bid 
                        0 - reviewer did not want to bid

            m: cuttoff value
        Returns:
            A float representing the precision
        """

        topM = np.asarray(ranked_list)[:m] != 0
        return np.mean(topM)

    def graph_precision_values(self, precision_values):
        """ Graph the recall values against M values """
        fig, ax = plt.subplots()
        df_recall = pd.DataFrame({
                '@M': range(1, len(precision_values)+1),
                'Recall': precision_values
            })

        ax = df_recall.plot.line(x="@M", y="Recall", ax=ax)
        ax.set_title("Recall Curve", y=1.08)
        ax.set_ylabel("Recall")
        fig.savefig("results/figures/{0}".format("recall_curve_bow_avg"), dpi=200)