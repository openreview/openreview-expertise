import argparse
import os
from . import utils


def data_to_sample(data, vocab, max_num_keyphrases):
    """
    Converts one line of the training data into a training sample.

    Training samples consist of the following:

    source:
        a numpy array containing integers. Each integer corresponds to
        a token in the vocabulary. This array of tokens represents the
        source text.
    source_length:
        a list containing one element, an integer, which is the number
        of keyphrases in 'source'.
    positive:
        ...
    positive_length:
        Similar to "source_length", but applies to the "positive" list.
    negative:
        ...
    negative_length:
        Similar to "source_length", but applies to the "negative" list.

    """

    source = vocab.to_ints(data["source"], max_num_keyphrases=max_num_keyphrases)
    source_length = [len(source)]
    positive = vocab.to_ints(data["positive"], max_num_keyphrases=max_num_keyphrases)
    positive_length = [len(positive)]
    negative = vocab.to_ints(data["negative"], max_num_keyphrases=max_num_keyphrases)
    negative_length = [len(negative)]
    sample = {
        "source": source,
        "source_length": source_length,
        "source_id": data["source_id"],
        "positive": positive,
        "positive_length": positive_length,
        "positive_id": data["positive_id"],
        "negative": negative,
        "negative_length": negative_length,
        "negative_id": data["negative_id"],
    }
    return sample


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('config_path')
#     parser.add_argument('datafile')
#     parser.add_argument('--samples_filename', default='train_samples_permuted.jsonl')
#     args = parser.parse_args()

#     config_path = os.path.abspath(args.config_path)
#     config = Config(args.config_path)

#     vocab = config.setup_load('vocab.pkl')
#     data_reader = utils.jsonl_reader(args.datafile)
#     train_samples = (data_to_sample(data, vocab) for data in data_reader)
#     config.setup_save(train_samples, args.samples_filename)
