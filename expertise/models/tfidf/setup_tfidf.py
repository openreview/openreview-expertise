import os
import argparse
import expertise

default_config = {
    'name': '',
    'dataset': {
        'directory': ''
    },
    'experiment_dir': './',
    'min_count_for_vocab': 1,
    'num_processes': 4,
    'random_seed': 9,
    'max_num_keyphrases': 25,
    'max_seq_length': 512,
    'do_lower_case': True,
    'embedding_aggregation_type': 'all',
    'batch_size': 32,
    'use_cuda': False,
    'kp_setup_dir': '',
    'tfidf_model': ''
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir')
    parser.add_argument('-c', '--config_path', default='./config-tfidf.json', help="a config file for a model")
    args = parser.parse_args()

    if args.dataset_dir:
        default_config.update({'dataset': {'directory': args.dataset_dir}})

    config = expertise.config.ModelConfig()
    config.update(**default_config)
    config.save(args.config_path)

    config.update_from_file(args.config_path)
    config.save(args.config_path)

    print(config)

    dataset = expertise.dataset.Dataset(**config.dataset)

    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)
    config.update(setup_dir=setup_dir)
    config.save(args.config_path)

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = expertise.utils.split_ids(list(dataset.submission_ids), seed=config.random_seed)

    bids_by_forum = expertise.utils.get_bids_by_forum(dataset)

    test_labels = expertise.utils.format_bid_labels(test_set_ids, bids_by_forum)

    expertise.utils.dump_jsonl(os.path.join(setup_dir, 'test_labels.jsonl'), test_labels)

