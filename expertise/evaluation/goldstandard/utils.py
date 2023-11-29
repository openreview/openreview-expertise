# Auxiliary functions to streamline the similarity computation.

import pandas as pd
from itertools import combinations
import numpy as np

def to_dicts(df):
    """Process the released dataframe in .csv format into a dict of dicts that is used in the
    evaluation and prediction stages.

    Args:
        df: released dataset in csv format.

    Returns:
        refs: reference similarities.
         {reviewer1: {paper1: expertise1, paper2: expertise2, ...}, ...}

        targets: Dict of sets. Each key in this dict corresponds to a reviewer and each
        set is a set of papers for which the reviewer evaluated their expertise.
         {reviewer1: {paper1, paper2, paper3, ... }, ...}
    """
    refs, targets = {}, {}

    for idx, row in df.iterrows():
        key = str(row['ParticipantID'])

        refs[key] = {row[f'Paper{x}']: row[f'Expertise{x}'] for x in range(1, 11)
                     if not pd.isna(row[f'Paper{x}'])}

        targets[key] = set([row[f'Paper{x}'] for x in range(1, 11)
                            if not pd.isna(row[f'Paper{x}'])])

    return refs, targets

def compute_main_metric(preds, refs, vp, vr):
    """Compute accuracy of predictions against references (weighted kendall's tau metric)

    Args:
        preds: dict of dicts, where top-level keys corresponds to reviewers
        and inner-level keys correspond to the papers associated with a given
        reviewer in the dataset. Values in the inner dicts should represent similarities
        and must be computed for all (valid_reviewer, valid_paper) pairs from the references.

        refs: ground truth values of reviewer expertise. The structure of the object
        is the same as that of preds.

        vp: papers to use in evaluations
        vr: reviewers to use in evaluations

    Returns:
        Loss of predictions.

    Note: Absolute values of *predicted* similarities do not matter, only the ordering is used to
    compute the score. Values of similarities in the references are used to weight mistakes.
    """

    max_loss, loss = 0, 0

    for reviewer in vr:

        papers = list(refs[reviewer].keys())

        for p1, p2 in combinations(papers, 2):

            if p1 not in vp or p2 not in vp:
                continue

            pred_diff = preds[reviewer][p1] - preds[reviewer][p2]
            true_diff = refs[reviewer][p1] - refs[reviewer][p2]

            max_loss += np.abs(true_diff)

            if pred_diff * true_diff == 0:
                loss += np.abs(true_diff) / 2

            if pred_diff * true_diff < 0:
                loss += np.abs(true_diff)

    return loss / max_loss


def compute_resolution(preds, refs, vp, vr, regime='easy'):
    """Compute resolution ability of the algorithms for easy/hard pairs of papers.

    Args:
        preds: dict of dicts, where top-level keys corresponds to reviewers
        and inner-level keys correspond to the papers associated with a given
        reviewer in the dataset. Values in the inner dicts should represent similarities
        and must be computed for all (valid_reviewer, valid_paper) pairs from the references.

        refs: ground truth values of reviewer expertise. The structure of the object
        is the same as that of predictions.

        vp: papers to use in evaluations
        vr: reviewers to use in evaluations

        regime: whether to score resolution for hard cases (two papers with score 4+)
        or easy papers (one paper with score 4+, one paper with score 2-)

    Returns:
        Dictionary capturing the loss of predictions.

    Note: Absolute values of *predicted* similarities do not matter, only the ordering is used to
    compute the score. Each mistake costs 1 (we do not weigh by delta between similarities).
    """

    if regime not in {'easy', 'hard'}:
        raise ValueError("Wrong value of the argument ('regime')")

    num_pairs = 0
    num_correct = 0

    for reviewer in vr:

        papers = list(refs[reviewer].keys())

        for p1, p2 in combinations(papers, 2):

            if p1 not in vp or p2 not in vp:
                cnt_bounds += 1
                continue

            s1 = refs[reviewer][p1]
            s2 = refs[reviewer][p2]

            # We only look at pairs of papers that are not tied in terms of the expertise
            if s1 == s2:
                continue

            # Hard-coded parameters to define HARD pairs
            if regime == 'hard' and min(s1, s2) < 4:
                continue

            # Hard-coded parameters to define EASY pairs
            if regime == 'easy' and (max(s1, s2) < 4 or min(s1, s2) > 2):
                continue

            num_pairs += 1
            pred_diff = preds[reviewer][p1] - preds[reviewer][p2]
            true_diff = s1 - s2

            # An algorithm is correct if the ordering of predicted similarities agrees
            # with the ordering of the ground-truth expertise
            if pred_diff * true_diff > 0:
                num_correct += 1

    # It may be the case that no pairs from a regime are present
    if num_pairs <= 0:
        return {'score': 0, 'correct': 0, 'total': 0}

    return {'score': num_correct / num_pairs, 'correct': num_correct, 'total': num_pairs}

def score_resolution(predictions, references, valid_papers, bootstraps=None):
    """Compute resolution ability of the algorithms (accuracy on hard/easy triples) together with bootstrapped
    values for confidence intervals

    :param predictions: JSON where predicted similarities are stored
    :param references: Ground truth values of expertise
    :param valid_papers: Papers to include in evaluation
    :param bootstraps: Subsampled reviewer pools for bootstrap computations
    :return: Score of predictions on easy/hard triples + data to compute confidence intervals (if `bootstraps` is not None)
    """

    valid_reviewers = list(predictions.keys())

    score_easy = compute_resolution(predictions, references, valid_papers, valid_reviewers, regime='easy')
    score_hard = compute_resolution(predictions, references, valid_papers, valid_reviewers, regime='hard')

    if bootstraps is None:
        return score_easy, score_hard

    variations_easy = [compute_resolution(predictions, references, valid_papers, vr, regime='easy')['score']
                                for vr in bootstraps]
    variations_hard = [compute_resolution(predictions, references, valid_papers, vr, regime='hard')['score']
                                for vr in bootstraps]

    return score_easy, score_hard, variations_easy, variations_hard
