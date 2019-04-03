import os
import importlib
from collections import defaultdict
from expertise.utils import dump_pkl, partition
from expertise.utils.dataset import Dataset
from tqdm import tqdm
import gensim

from expertise.models.bert import feature_extractor

def write_bert_data(filename, text, max_seq_length=None):
    if not max_seq_length:
        sentences = gensim.summarization.textcleaner.split_sentences(text)
    else:
        sentences = split_by_chunks(text, max_seq_length)

    with open(filename, 'w') as f:
        for sentence in sentences:
            f.write(sentence)
            f.write('\n')

def split_by_chunks(text, max_seq_length):
    sentences = gensim.summarization.textcleaner.split_sentences(text)
    chunks = [[]]
    for sentence in sentences:
        for word in sentence.split():
            if len(chunks[-1]) >= max_seq_length:
                chunks.append([])

            chunks[-1].append(word)
    chunked_sentences = [' '.join(chunk) for chunk in chunks]
    return chunked_sentences

def setup(config, partition_id=0, num_partitions=1):

    bert_base_dir = config.bert_base_dir
    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    submissions_dir = os.path.join(setup_dir, 'submissions')
    if not os.path.exists(submissions_dir):
        os.mkdir(submissions_dir)

    archives_dir = os.path.join(setup_dir, 'archives')
    if not os.path.exists(archives_dir):
        os.mkdir(archives_dir)

    submissions_features_dir = os.path.join(setup_dir, 'submissions-features')
    if not os.path.exists(submissions_features_dir):
        os.mkdir(submissions_features_dir)

    archives_features_dir = os.path.join(setup_dir, 'archives-features')
    if not os.path.exists(archives_features_dir):
        os.mkdir(archives_features_dir)

    dataset = Dataset(**config.dataset)

    for filename, text in tqdm(dataset.submissions(fields=['title','abstract']), total=dataset.num_submissions, desc='parsing submission keyphrases'):
        new_filename = '{}.txt'.format(filename.replace('.jsonl', ''))
        new_filepath = os.path.join(submissions_dir, new_filename)
        if not os.path.exists(new_filepath):
            write_bert_data(new_filepath, text)

    for file_idx, (filename, text) in enumerate(tqdm(dataset.archives(fields=['title','abstract']), total=dataset.num_archives, desc='parsing archive keyphrases')):
        file_id = filename.replace('.jsonl', '')
        new_filename = '{:05d}|{}.txt'.format(file_idx, file_id)
        new_filepath = os.path.join(archives_dir, new_filename)
        if not os.path.exists(new_filename):
            write_bert_data(new_filepath, text)

    submission_files = list(partition(
        sorted(os.listdir(submissions_dir)),
        partition_id=int(partition_id),
        num_partitions=int(num_partitions)
    ))

    for file in tqdm(submission_files, total=len(submission_files), desc='extracting submission features'):
        input_file = os.path.join(submissions_dir, file)
        output_file = os.path.join(submissions_features_dir, file)
        if not os.path.exists(output_file):
            feature_extractor.extract(
                input_file=input_file,
                vocab_file=os.path.join(bert_base_dir, 'vocab.txt'),
                bert_config_file=os.path.join(bert_base_dir, 'bert_config.json'),
                init_checkpoint=os.path.join(bert_base_dir, 'bert_model.ckpt'),
                output_file=output_file,
                max_seq_length=config.max_seq_length
            )

    archives_files = list(partition(
        sorted(os.listdir(archives_dir)),
        partition_id=int(partition_id),
        num_partitions=int(num_partitions)
    ))

    for file in tqdm(archives_files, total=len(archives_files), desc='extracting archive features'):
        input_file = os.path.join(archives_dir, file)
        output_file = os.path.join(archives_features_dir, file)
        if not os.path.exists(output_file):
            feature_extractor.extract(
                input_file=input_file,
                vocab_file=os.path.join(bert_base_dir, 'vocab.txt'),
                bert_config_file=os.path.join(bert_base_dir, 'bert_config.json'),
                init_checkpoint=os.path.join(bert_base_dir, 'bert_model.ckpt'),
                output_file=output_file,
                max_seq_length=config.max_seq_length
            )

    #reviewer_kps_path = os.path.join(setup_dir, 'reviewer_kps.pkl')
    #dump_pkl(reviewer_kps_path, kps_by_reviewer)

def train(config):
    pass

def infer(config):
    pass

def test(config):
    pass
