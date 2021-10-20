import argparse
import itertools
from pathlib import Path

import numpy as np

import expertise
from expertise import utils
from expertise.dataset import Dataset


def infer(config):
    experiment_dir = Path(config["experiment_dir"]).resolve()

    model = utils.load_pkl(config["tfidf_model"])

    dataset = Dataset(**config["dataset"])

    paperids = list(model.bow_archives_by_paperid.keys())
    paperidx_by_id = {paperid: index for index, paperid in enumerate(paperids)}

    score_file_path = experiment_dir.joinpath(config["name"] + "-scores.csv")

    bids_by_forum = expertise.utils.get_bids_by_forum(dataset)
    submission_ids = [n for n in dataset.submission_ids]
    reviewer_ids = [r for r in dataset.reviewer_ids]
    # samples = expertise.utils.format_bid_labels(submission_ids, bids_by_forum)

    scores = {}
    max_score = 0.0
    for paperid, userid in itertools.product(submission_ids, reviewer_ids):
        # label = data['label']

        if userid not in scores:
            # bow_archive is a list of BOWs.
            if (
                userid in model.bow_archives_by_userid
                and len(model.bow_archives_by_userid[userid]) > 0
            ):
                bow_archive = model.bow_archives_by_userid[userid]
            else:
                bow_archive = [[]]

            best_scores = np.amax(model.index[bow_archive], axis=0)
            scores[userid] = best_scores

            user_max_score = max(best_scores)
            if user_max_score > max_score:
                max_score = user_max_score

    print("max score", max_score)

    with open(score_file_path, "w") as w:
        for userid, user_scores in scores.items():
            for paperidx, paper_score in enumerate(user_scores):
                paperid = paperids[paperidx]
                score = scores[userid][paperidx] / max_score

                w.write("{0},{1},{2:.3f}".format(paperid, userid, score))
                w.write("\n")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="a config file for a model")
    args = parser.parse_args()

    config = expertise.config.ModelConfig()
    config.update_from_file(args.config_path)

    infer(config)
