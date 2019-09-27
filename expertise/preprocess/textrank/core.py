
import itertools
import os
from tqdm import tqdm

from expertise.utils.vocab import Vocab
from expertise.dataset import Dataset
import expertise.utils as utils

from .textrank_words import keyphrases

def run_textrank(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''

    experiment_path = os.path.dirname(config.experiment_dir)

    kps_dir = os.path.join(experiment_path, 'keyphrases')
    if not os.path.isdir(kps_dir):
        os.makedirs(kps_dir)
    config.update(kp_setup_dir=kps_dir)

    print('starting setup')
    dataset = Dataset(directory=config.dataset['directory'])
    textrank_vocab = Vocab() # vocab used for textrank-based keyphrases
    full_vocab = Vocab() # vocab used on the full text

    print('keyphrase extraction')
    textrank_kps_by_id = {}
    full_kps_by_id = {}

    all_archives = itertools.chain(
        dataset.submissions(return_batches=True),
        dataset.archives(return_batches=True))

    for archive_id, content_list in tqdm(
            all_archives, total=dataset.total_archive_count + dataset.submission_count):

        scored_kps = []
        full_kps = []
        for content in content_list:
            text = utils.content_to_text(content)
            top_tokens, full_tokens = keyphrases(text, include_scores=True, include_tokenlist=True)
            scored_kps.extend(top_tokens)
            full_kps.append(full_tokens)
        sorted_kps = [kp for kp, _ in sorted(scored_kps, key=lambda x: x[1], reverse=True)]

        top_kps = []
        kp_count = 0
        for kp in sorted_kps:
            if kp not in top_kps:
                top_kps.append(kp)
                kp_count += 1
            if kp_count >= config.max_num_keyphrases:
                break

        textrank_vocab.load_items(top_kps)
        full_vocab.load_items([kp for archive in full_kps for kp in archive])
        assert archive_id not in textrank_kps_by_id
        textrank_kps_by_id[archive_id] = top_kps
        full_kps_by_id[archive_id] = full_kps

    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'textrank_kps_by_id.pkl'), textrank_kps_by_id)
    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'full_kps_by_id.pkl'), full_kps_by_id)
    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'textrank_vocab.pkl'), textrank_vocab)
    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'full_vocab.pkl'), full_vocab)

    return config
