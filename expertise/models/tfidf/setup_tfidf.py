import os
import argparse
import expertise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    config = expertise.config.ModelConfig()
    config.update_from_file(args.config_path)

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

