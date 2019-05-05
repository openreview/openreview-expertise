import os
import importlib
import itertools
from collections import defaultdict
from expertise.utils import dump_pkl
from expertise.utils.dataset import Dataset
from expertise.preprocessors.textrank import TextRank
from expertise.preprocessors.pos_regex import extract_candidate_words
from expertise.preprocessors.pos_regex import extract_candidate_chunks
from expertise.preprocessors.pos_regex_spacy import keywords as spacy_keywords
from expertise.preprocessors.pos_regex_spacy import chunks as spacy_chunks
from expertise.utils.standard_setup import setup_train_dev_test
from tqdm import tqdm

import ipdb

def get_flat_expansion(text, config):
    textrank = TextRank()
    textrank.analyze(text)
    expanded_kps = [
        [k] * textrank.counts[k]
        for k, _
        in textrank.keyphrases()
    ]
    return [k for l in expanded_kps for k in l]

def setup(config):
    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    train_split, dev_split, test_split = setup_train_dev_test(config)

    dataset = Dataset(**config.dataset)

    # get submission contents
    # formerly "paper_content_by_id"

    kps_by_submission = defaultdict(list)
    for file_id, text_list in dataset.submissions():
        for text in text_list:
            # keyphrases = get_flat_expansion(text, config)
            keyphrases = spacy_chunks(text)
            kps_by_submission[file_id].extend(keyphrases)

    submission_kps_path = os.path.join(setup_dir, 'submission_kps.pkl')
    dump_pkl(submission_kps_path, kps_by_submission)

    # write keyphrases for reviewer archives to pickle file
    # formerly "reviewer_content_by_id"
    kps_by_reviewer = defaultdict(list)

    for file_id, text_list in dataset.archives():
        for text in text_list:
            # keyphrases = get_flat_expansion(text, config)
            keyphrases = spacy_chunks(text)
            kps_by_reviewer[file_id].append(keyphrases)

    reviewer_kps_path = os.path.join(setup_dir, 'reviewer_kps.pkl')
    dump_pkl(reviewer_kps_path, kps_by_reviewer)
