# Paper-reviewer affinity modeling for OpenReview

A key part of matching papers to reviewers is having a good model of paper-reviewer affinity. This repository holds code and tools for generating affinity scores between papers and reviewers.

### Installation & Getting Started

Clone this repository and install the package using pip as follows:

```
pip install <location of this repository>
```

If you plan to actively develop models, it's best to install the package in "edit" mode, so that you don't need to reinstall the package every time you make changes:

```
pip install -e <location of this repository>
```

The framework requires a valid OpenReview Dataset (see Dataset section below). Contact Michael for access to datasets.

### Workflow

There are four stages of the workflow:

`(1) Setup --> (2) Train --> (3) Infer --> (4) Test`

Each model in `expertise.models` is expected to implement functions for each of these stages. Models will usually output intermediate files after each stage for use in the next stage(s).

The framework doesn't enforce any of the suggested stages; they're provided for organizational guidance only. The behavior at each stage is model-specific, and models are responsible for managing their own inputs and outputs at each stage. The following guidelines are provided to maintain general organization:

**Setup**:
Performs any necessary preprocessing on the dataset.
```
example:
python -m expertise.setup_model config.json
```

**Train**:
Trains the model, if applicable.
```
example:
python -m expertise.train_model config.json
```

**Infer**:
Produces scores for every paper-reviewer pair in the dataset. The output of this stage can be used by the OpenReview Matching System.
```
example:
python -m expertise.infer_model config.json
```

**Test**:
Evaluates the performance of the model. Can be performed either on the inferred scores on the entire dataset or on a selected testing subset.
```
example:
	python -m expertise.test_model config.json
```

### Configuration
Models are driven by a configuration JSON file, usually located in an "experiment directory". Configurations are expected to have the following properties:

1) `name`: a string that identifies the experiment (avoid using spaces in this field).
2) `experiment_dir`: a string that identifies the experiment's location
3) `dataset`: a string representing the directory where the dataset is located.
4) `model`: a string that specifies the model module to be trained (from `expertise.models`).

All other attributes in the config file are specific to the type of model and experiment being run. The example below shows what a configuration for the TF-IDF model could look like:

```
{
    "name": "midl19-tfidf",
    "dataset": {
        "directory": "/path/to/midl19/dataset"
    },
    "experiment_dir": "/path/to/midl19/experiment",
    "model": "expertise.models.tfidf",
    "keyphrases": "expertise.preprocessors.pos_regex",
    "max_num_keyphrases": 10,
    "min_count_for_vocab": 1,
    "num_processes": 4,
    "random_seed": 2524
}
```

All of the workflow stages expect this configuration file as input.

### Datasets

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

See `expertise.utils.dataset` for implementation details.

