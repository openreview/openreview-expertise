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

The framework requires a valid OpenReview Dataset (see Dataset section below). Contact Michael for access to datasets.


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
	extras/
		bids.jsonl
		(other dataset-specific files)
		...
	README.md

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

