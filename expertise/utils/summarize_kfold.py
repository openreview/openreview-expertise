"""
This script should summarize the results of an experiment across K folds
"""
import argparse
import os
from expertise.config import ModelConfig
import random
import ipdb
import csv
import numpy as np


def get_scores(config, k):
    old_experiment_dir = config.experiment_dir
    new_experiment_dir = os.path.join(old_experiment_dir, f"{config.name}{k}")

    data = {}
    with open(os.path.join(new_experiment_dir, "test", "test.scores.tsv")) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            data[row[1]] = float(row[2])

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("num_folds", type=int)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    experiment_path = os.path.dirname(config_path)

    config = ModelConfig()
    config.update_from_file(config_path)

    all_data = []
    for k in range(args.num_folds):
        all_data.append(get_scores(config, k))

    summary = {}
    for measure in ["MAP", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]:
        summary[measure] = np.mean([d[measure] for d in all_data])

    with open(
        os.path.join(config.experiment_dir, f"{config.name}_summary.csv"), "w"
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        for measure, value in summary.items():
            writer.writerow([config.name, measure, value])
