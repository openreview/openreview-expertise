# Paper-reviewer affinity modeling for OpenReview

A key part of matching papers to reviewers is having a good model of paper-reviewer affinity. This repository holds code and tools for generating affinity scores between papers and reviewers.

## Installation & Getting Started

Clone this repository and install the package using pip as follows:

```
pip install <location of this repository>
```

If you plan to actively develop models, it's best to install the package in "edit" mode, so that you don't need to reinstall the package every time you make changes:

```
pip install -e <location of this repository>
```

## Quick Start

Start by creating an "experiment directory" (`experiment_dir`), and a JSON config file (e.g. `config.json`) in it.

Example configuration for the TF-IDF Sparse Vector Similarity model:
```
{
    "name": "iclr2020_reviewers_tfidf",
    "match_group": "ICLR.cc/2020/Conference/Reviewers",
    "paper_invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
    "exclusion_inv": "ICLR.cc/2020/Conference/-/Expertise_Selection",
    "min_count_for_vocab": 1,
    "random_seed": 9,
    "max_num_keyphrases": 25,
    "do_lower_case": true,
    "dataset": {
        "directory": "./"
    },
    "experiment_dir": "./"
}

```

Create a dataset by running the following command:
```
python -m expertise.create_dataset config.json \
	--baseurl <usually https://openreview.net> \
	--password <your_password> \
	--username <your_username>\
```

Generate scores by running the following command:
```
python -m expertise.tfidf_scores config.json
```

The output will generate a `.csv` file with the name pattern `<config_name>-scores.csv`.


## Configuration

**Coming Soon**

## Datasets

The framework expects datasets to adhere to a specific format. Each dataset directory should be structured as follows:

```
dataset-name/
	archives/
		~User_Id1.jsonl 		# user's tilde IDs
		~User_Id2.jsonl
		...
	submissions/
		aBc123XyZ.jsonl 		# paper IDs
		ZYx321Abc.jsonl
		...
	bids/
		aBc123XyZ.jsonl 		# should have same IDs as /submissions
		ZYx321Abc.jsonl
		...

```

Submissions are one-line .jsonl files containing the paper's content field. Archives are .jsonl files with any number of lines, where each line contains the content field of a paper authored by that user.

Some datasets differ slightly in terms of the format of the data; these should be accounted for in the experiment's configuration.

**For example**: some older conferences use a bidding format that differs from the default "Very High" to "Very Low" scale. This can be parameterized in the `config.json` file (e.g.) as follows:

```
{
    "name": "uai18-tfidf",
    "dataset": {
        "directory": "/path/to/uai18/dataset",
        "bid_values": [
            "I want to review",
            "I can review",
            "I can probably review but am not an expert",
            "I cannot review",
            "No bid"
        ],
        "positive_bid_values": ["I want to review", "I can review"]
    },
    ...
}

```

See `expertise.utils.dataset` for implementation details.

