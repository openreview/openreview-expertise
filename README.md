# Paper-reviewer affinity modeling for OpenReview

A key part of matching papers to reviewers is having a good model of paper-reviewer affinity. This repository holds code and tools for generating affinity scores between papers and reviewers.

## Installation

This repository only supports Python 3.6 and above.
Clone this repository and install the package using pip as follows. If you plan to use ELMo, then you will need to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), since one of the packages is only available in conda. You may use the `pip` command in a conda environment as long as you first run all the pip installs and then conda installs. Just follow the order of the commands shown below and it should work. You may read more about this [here](https://www.anaconda.com/using-pip-in-a-conda-environment/).

Run this command only if you are using conda:
```
conda create -n affinity python=3.7
conda activate affinity
conda install pip
```

```
pip install <location of this repository>
```

If you plan to actively develop models, it's best to install the package in "edit" mode, so that you don't need to reinstall the package every time you make changes:

```
pip install -e <location of this repository>
```

Because some of the libraries are specific to our operating system you would need to install these dependencies separately. We expect to improve this in the future. If you plan to use ELMo with GPU you need to install [pytorch](https://pytorch.org/) by selecting the right configuration for your particular OS, otherwise, if you are only using the CPU, the current dependencies should be fine. We also use [faiss](https://github.com/facebookresearch/faiss/) for ELMo to calculate vector similarities. This is not included in the dependencies inside `setup.py` because the official package is only available in conda.

Run this command if you plan to use ELMo (Using CPU is fine):
```
conda install intel-openmp==2019.4
conda install faiss-cpu -c pytorch
```
[Here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) you can find the above installation command.

If you plan to use SPECTER, follow the set-up steps 1 and 2 described [here](https://github.com/allenai/specter#how-to-use-the-pretrained-model). Provide the SPECTER repository path in the config as `model_params.specter_dir`.

If you plan to use Multifacet-Recommender, download the checkpoint files from [here](https://drive.google.com/file/d/1_mWkQ1dr_Vl121WZkbNyNMV3G_bmoQ6s/view?usp=sharing) and pass the path to `feature_vocab_file` and `model_checkpoint_dir`

## Affinity Scores

There are two steps to create affinity scores:
- Create Dataset
- Run Model

The dataset can be generated using the [OpenReview python API](https://github.com/openreview/openreview-py) which should be installed when this repository is installed. You can generate your own dataset from some other source as long as it is compliant with the format shown in the Datasets section.
Start by creating an "experiment directory" (`experiment_dir`), and a JSON config file (e.g. `config.json`) in it. Go to the Configuration File section for details on how to create the `config.json`.

Create a dataset by running the following command:
```
python -m expertise.create_dataset config.json \
	--baseurl <usually https://openreview.net> \
	--password <your_password> \
	--username <your_username> \
```

For SPECTER, ELMo and BM25 run the following command
```
python -m expertise.run config.json
```
The output will generate a `.csv` file with the name pattern `<config_name>.csv`.

For TF-IDF Sparse Vector Similarity run the following command:
```
python -m expertise.tfidf_scores config.json
```

The output will generate a `.csv` file with the name pattern `<config_name>-scores.csv`.

## Detect Duplicates

Duplicate detection can be used to find duplicates both within a venue or between 2 different venues.

- For detecting duplicates within a venue only one submissions directory or one submissions.jsonl file is needed.
- For detecting duplicates between 2 different venues, either one submissions and one other submissions directories are needed or one submissions.jsonl and one other_submissions.jsonl are needed.

More details about these files/directories can be found at the end of this section.

There are two steps to detect duplicates:
- Create Dataset or Datasets
- Run Duplicate Detection

The dataset can be generated using the [OpenReview python API](https://github.com/openreview/openreview-py) which should be installed when this repository is installed. You can generate your own dataset from some other source as long as it is compliant with the format shown in the Datasets section.
Start by creating an "experiment directory" (`experiment_dir`), and a JSON config file (e.g. `config.json`) in it. Go to the Configuration File section for details on how to create the `config.json`.

Duplicate detection uses ELMo exclusively, since we always normalize the scores for BM25. ELMo scores have values from 0 to 1. The closer a score is to 1, the more similar the submissions are. The `normalize` option for ELMo is disabled for duplicate detection.

Create a dataset by running the following command (this is optional if you already have the dataset):
```
python -m expertise.create_dataset config.json \
	--baseurl <usually https://openreview.net> \
	--password <your_password> \
	--username <your_username> \
```

For ELMo run the following command
```
python -m expertise.run_duplicate_detection config.json
```
The output will generate a `.csv` file with the name pattern `<config_name>.csv`. Read the `Configuration File` section to understand how to create one. For duplicate detection, the parameters that apply are in `Affinity Scores Configuration Options`, `ELMo specific parameters (affinity scores)`, and `ELMo specific parameters (duplicate detection)`.

## Configuration File

The configuration file or `config.json` is the file that contains all the parameters to calculate affinity scores.
Below you will find examples of possible configurations depending on the Model that you want to use. You may have a config file for creating the dataset and another for generating the affinity scores, something like `dataset-config.json` and `affinity-config.json`. However you can have everything in a single file like in the examples below:

### Create Dataset Configuration Options
This parameters could be included in a separate file, like `dataset-config.json`, as was mentioned before.
- `match_group`: (optional) String or array of strings containing the groups of Reviewers or Area Chairs. The Reviewers (and Area Chairs) will get affinity scores with respect to the submitted papers based on their expertise. This expertise is obtained based on the publications available in OpenReview. It can be provided instead of or on top of the `reviewer_ids` and `csv_expertise`, the values will be combined.
- `reviewer_ids`: (optional) Array of strings containing the IDs of the reviewers. An ID can either be an email associated to an OpenReview Profile or the OpenReview ID, also known as a Tilde ID, for a Profile. The array can be a combination of both these types of IDs. It can be provided instead of or on top of the `match_group` and `csv_expertise`, the values will be combined.
- `csv_expertise`: (optinal) String with the relative path from `dataset.directory` to a csv file containing the expertise in the following format: "author id","publication_id","title","abstract". This can be added instead of or on top of `match_group` and `reviewer_ids`, the values will be combined.
- `paper_invitation`: (optional) String or array of strings with the submission invitations. This is the invitation for Submissions, all the submissions in OpenReview for a particular venue have an invitation and that is how they are grouped together.
- `exclusion_inv` (optional): String or array of strings with the exclusion invitations. Reviewers (and Area Chairs) can choose to exclude some of their papers before the affinity scores are calculated so that they get papers that are more aligned to their current expertise/interest. Papers included here will not be taken into consideration when calculating the affinity scores.
- `csv_submissions`: (optional) String with the relative path from `dataset.directory` to a csv file containing the submissions in the following format: "submission_id","title","abstract". This can be added on top of `paper_invitation`, the values will be combined.
- `bid_inv` (optional): String or array of strings with the bid invitations. Bids are used by the reviewers in OpenReview to select papers that they would or would not like to review. These bids are then used to compute a final affinity score to be more fair with the reviewers.
- `use_email_ids` (optional): Boolean value. If true, then the email of the user is used instead of his/her OpenReview Profile ID.
- `max_workers` (optional): Number indicating the amount of workers that will be used to retrieve the expertise from OpenReview. If it is not set, it will use the maximum available workers of your machine by default. The more workers the faster the creation of the dataset
- `dataset.directory`: This is the directory where the data will be dumped. Once `create_dataset` finishes running, all the folders with the files inside will be in there.
- `dataset.minimum_pub_date` (optional): Number indicating the Unix date in milliseconds (that's what we use in OpenReview) of the publication. Any publication before this date will not be included in the dataset. If this parameter is included with `dataset.top_recent_pubs`, then, the intersection of the publications meeting both criteria will be selected. If instead, the user wants to include the union between both results, then, both parameters should be included inside the `dataset.or` field. Look at the examples for more details.
- `dataset.top_recent_pubs` (optional): Number or string indicating a percentage. If the user sets a number, like 3, then only the 3 most recent publications are taken into consideration. If the reviewer has less than 3 publications, then all his/her publications are taken into consideration. For percentages, if we select 10%, this will still work for 3 publications. 10% of 3 is 0.3, however, decimal values are always rounded to the next integer, so the result in this case would be 1. If this parameter is included with `dataset.minimum_pub_date`, then, the intersection of the publications meeting both criteria will be selected. If instead, the user wants to include the union between both results, then, both parameters should be included inside the `dataset.or` field. Look at the examples for more details.

Here is an example:
```
{
    "match_group": ["ICLR.cc/2020/Conference/Reviewers", "ICLR.cc/2020/Conference/Area_Chairs"],
    "paper_invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
    "exclusion_inv": "ICLR.cc/2020/Conference/-/Expertise_Selection",
    "bid_inv": "ICLR.cc/2020/Conference/-/Add_Bid",
    "dataset": {
        "directory": "./"
    }
}
```

Here is an example with `minimum_pub_date` and `top_recent_pubs` with AND relationship:
```
{
    "match_group": ["ICLR.cc/2020/Conference/Reviewers", "ICLR.cc/2020/Conference/Area_Chairs"],
    "paper_invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
    "exclusion_inv": "ICLR.cc/2020/Conference/-/Expertise_Selection",
    "bid_inv": "ICLR.cc/2020/Conference/-/Add_Bid",
    "dataset": {
        "directory": "./",
        "minimum_pub_date": 1483228800000,
        "top_recent_pubs": 5
    }
}
```

Here is an example with `minimum_pub_date` and `top_recent_pubs` with OR relationship:
```
{
    "match_group": ["ICLR.cc/2020/Conference/Reviewers", "ICLR.cc/2020/Conference/Area_Chairs"],
    "paper_invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
    "exclusion_inv": "ICLR.cc/2020/Conference/-/Expertise_Selection",
    "bid_inv": "ICLR.cc/2020/Conference/-/Add_Bid",
    "dataset": {
        "directory": "./",
        "or": {
            "minimum_pub_date": 1483228800000,
            "top_recent_pubs": "10%"
        }
    }
}
```

Here is an example with `reviewer_ids`:
```
{
    "match_group": ["ICLR.cc/2020/Conference/Reviewers", "ICLR.cc/2020/Conference/Area_Chairs"],
    "reviewer_ids": ["~Carlos_Mondra1", "mondra@email.com", "1234@email.com", ...]
    "paper_invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
    "exclusion_inv": "ICLR.cc/2020/Conference/-/Expertise_Selection",
    "bid_inv": "ICLR.cc/2020/Conference/-/Add_Bid",
    "dataset": {
        "directory": "./"
    }
}
```

Here is an example with `csv_submissions` and `csv_expertise`. In this case, both files should be placed here `./csv_expertise.csv` and here `./csv_submissions`.
```
{
    "match_group": ["ICLR.cc/2020/Conference/Reviewers", "ICLR.cc/2020/Conference/Area_Chairs"],
    "csv_expertise": "csv_expertise.csv",
    "reviewer_ids": ["~Carlos_Mondra1", "mondra@email.com", "1234@email.com", ...]
    "paper_invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
    "csv_submissions": "csv_submissions.csv",
    "exclusion_inv": "ICLR.cc/2020/Conference/-/Expertise_Selection",
    "bid_inv": "ICLR.cc/2020/Conference/-/Add_Bid",
    "dataset": {
        "directory": "./"
    }
}
```

### Affinity Scores Configuration Options
These parameters could be included in a separate file, like `affinity-config.json`, as was mentioned before.

- `name`: This is the name that the `.csv` file containing the affinity scores will have.
- `model_params.scores_path`: This is the directory where the `.csv` file with the scores will be dumped.
- `model_params.use_title`: Boolean that indicates whether to use the title for the affinity scores or not. If this is `true` and `model_params.use_abstract` is also `true`, then, whenever a Submission or Publication does not have an abstract, it will fallback to the title.
- `model_params.use_abstract`: Boolean that indicates whether to use the abstract for the affinity scores or not. If this is `true` and `model_params.use_title` is also `true`, then, whenever a Submission or Publication does not have an abstract, it will fallback to the title.
- `model_params.sparse_value` (optional): Numerical value. If passed, instead of returning all the possible reviewer-submission combinations, only the top scores will be returned. The number of top scores will be determined by the `sparse_value`. That does not mean that the number of scores per submission will be equal to the `sparse_value`. Here is an example, if there are only 10 submissions and 5 reviewers, there would be a total of 50 scores. If we set the `sparse_value` to 5, each reviewer will get the top 5 submissions that are most similar to their publications. However, there might be a submission (or more submissions) that is not among the top 5 of any reviewer. In order to ensure that all submissions also have reviewers, the top 5 reviewers are assigned to each submission. As you can imagine, some reviewers will have more than 5 submissions assigned because of this reason.
- `dataset.directory`: This is the directory where the data will be read from. If `create_dataset` is used, then the files will have the required format. If, however, the data does not come from OpenReview, then the dataset should be compliant with the format specified in the Datasets section.
- `dataset.with_title` (optional): Boolean to indicate if only publications in OpenReview with title should be included.
- `dataset.with_abstract` (optional): Boolean to indicate if only publications in OpenReview with abstract should be included.

#### BM25Okapi specific parameters:
- `model_params.workers`: This is the number of processes that for BM25Okapi. This depends on your machine, but 4 is usually a safe value.

Here is an example:
```
{
    "name": "iclr2020_bm25_abstracts",
    "dataset": {
        "directory": "./"
    },
    "model": "bm25",
    "model_params": {
        "scores_path": "./",
        "use_title": false,
        "use_abstract": true,
        "workers": 4,
        "publications_path": "./",
        "submissions_path": "./"
    }
}
```

#### ELMo specific parameters (affinity scores):
- `model_params.use_cuda`: Boolean to indicate whether to use GPU (`true`) or CPU (`false`) when running ELMo. Currently, only 1 GPU is supported, but there does not seem to be necessary to have more.
- `model_params.batch_size`: Batch size when running ELMo. This defaults to 8, but depending on your machine, this value could be different.
- `model_params.publications_path`: When running ELMo, this is where the embedded abstracts/titles of the Reviewers (and Area Chairs) are stored.
- `model_params.submissions_path`: When running ELMo, this is where the embedded abstracts/titles of the Submissions are stored.
- `model_params.knn` (optional): This parameter specifies the k Nearest Neighbors that will be printed to the csv file. For instance, if the value is 10, then only the first 10 authors with the highest affinity score will be printed for each submission. You may see that if the value is 10, more than 10 values are printed, that is because there are ties in the scores. If the parameter is not specified, then each submission will have a score for every reviewer.
- `model_params.normalize` (optional): This parameter specifies if the ELMo scores should be normalized. Normally, the ELMo scores are between 0.5 and 1. Therefore, normalizing the scores can provide better matching between reviewers and submissions. This of course would not change the order of the results, if reviewer 1 is better than reviewer 2 for a particular submission, this will still be true after normalizing the scores.
- `model_params.skip_elmo`: Since running ELMo can take a significant amount of time, the vectors are saved in `model_params.submissions_path` and `model_params.publications_path`. If you want to run other operations with these results, like changing the value of `model_params.knn`, you may do so without running ELMo again by setting `model_params.skip_elmo` to true. The pickle files will be loaded with all the vectors.

Here is an example:
```
{
    "name": "iclr2020_elmo_abstracts",
    "dataset": {
        "directory": "./"
    },
    "model": "elmo",
    "model_params": {
        "scores_path": "./",
        "use_title": false,
        "use_abstract": true,
        "use_cuda": true,
        "batch_size": 8,
        "normalize": true,
        "skip_elmo": false,
        "publications_path": "./",
        "submissions_path": "./"
    }
}
```

#### SPECTER specific parameters (affinity scores):
- `model_params.specter_dir`: Path to the unpacked SPECTER directory. The model checkpoint will be loaded relative to this directory.
- `model_params.work_dir`: When running SPECTER, this is where the intermediate files are stored.
- `model_params.use_cuda`: Boolean to indicate whether to use GPU (`true`) or CPU (`false`) when running SPECTER. Currently, only 1 GPU is supported, but there does not seem to be necessary to have more.
- `model_params.batch_size`: Batch size when running SPECTER. This defaults to 16.
- `model_params.publications_path`: When running SPECTER, this is where the embedded abstracts/titles of the Reviewers (and Area Chairs) are stored.
- `model_params.submissions_path`: When running SPECTER, this is where the embedded abstracts/titles of the Submissions are stored.
- `model_params.average_score` (boolean, defaults to `false`): This parameter specifies that the reviewer is assigned based on the average similarity of the submission to the authored publication embeddings. Exactly one of `model_params.average_score` and `model_params.max_score` must be `true`.
- `model_params.max_score` (boolean, defaults to `true`): This parameter specifies that the reviewer is assigned based on the max similarity of the submission to the authored publication embeddings. Exactly one of `model_params.average_score` and `model_params.max_score` must be `true`.
- `model_params.skip_specter`: Since running SPECTER can take a significant amount of time, the vectors are saved in `model_params.submissions_path` and `model_params.publications_path`. The jsonl files will be loaded with all the vectors.

Here is an example:
```
{
    "name": "iclr2020_specter",
    "dataset": {
        "directory": "./"
    },
    "model": "specter",
    "model_params": {
        "specter_dir": "../specter/",
        "work_dir": "./",
        "average_score": false,
        "max_score": true,
        "use_cuda": true,
        "batch_size": 16,
        "publications_path": "./",
        "submissions_path": "./",
        "scores_path": "./"
    }
}
```

#### Multifacet-Recommender specific parameters (affinity scores):
- `model_params.feature_vocab_file`: Path to the vocablary file for the text encoders
- `model_params.model_checkpoint_dir`: Path to the directory containing the trained Multifacet-Recommender model checkpoints
- `model_params.work_dir`: When running Multifacet-Recommender, this is where the intermediate files are stored.
- `model_params.epochs`: Number of epochs to finetune reviewer embeddings. This defaults to 100.
- `model_params.batch_size`: Batch size when running Multifacet-Recommender. This defaults to 50.
- `model_params.use_cuda`: Boolean to indicate whether to use GPU (`true`) or CPU (`false`) when running Multifacet-Recommender. Currently, only 1 GPU is supported, but there does not seem to be necessary to have more.

Here is an example:
```
{
    "name": "iclr2020_mfr",
    "dataset": {
        "directory": "./"
    },
    "model": "mfr",
    "model_params": {
        "feature_vocab_file": "../mfr_data/feature_vocab_file",
        "model_checkpoint_dir": "../mfr_data/mfr_model_checkpoint/",
        "work_dir": "./",
        "epochs": 100,
        "batch_size": 50,
        "use_cuda": true,
        "scores_path": "./"
    }
}
```

#### SPECTER+MFR ensemble specific parameters (affinity scores):
- `model_params.specter_dir`: Path to the unpacked SPECTER directory. The model checkpoint will be loaded relative to this directory.
- `model_params.specter_batch_size`: Batch size when running SPECTER. This defaults to 16.
- `model_params.publications_path`: When running SPECTER, this is where the embedded abstracts/titles of the Reviewers (and Area Chairs) are stored.
- `model_params.submissions_path`: When running SPECTER, this is where the embedded abstracts/titles of the Submissions are stored.
- `model_params.average_score` (boolean, defaults to `false`): This parameter specifies that the reviewer is assigned based on the average similarity of the submission to the authored publication embeddings. Exactly one of `model_params.average_score` and `model_params.max_score` must be `true`.
- `model_params.max_score` (boolean, defaults to `true`): This parameter specifies that the reviewer is assigned based on the max similarity of the submission to the authored publication embeddings. Exactly one of `model_params.average_score` and `model_params.max_score` must be `true`.
- `model_params.skip_specter`: Since running SPECTER can take a significant amount of time, the vectors are saved in `model_params.submissions_path` and `model_params.publications_path`. The jsonl files will be loaded with all the vectors.
- `model_params.mfr_feature_vocab_file`: Path to the vocablary file for the Multifacet-Recommender text encoders.
- `model_params.mfr_checkpoint_dir`: Path to the directory containing the trained Multifacet-Recommender model checkpoints.
- `model_params.mfr_epochs`: Number of epochs to finetune reviewer embeddings for Multifacet-Recommender. This defaults to 100.
- `model_params.mfr_batch_size`: Batch size when running Multifacet-Recommender. This defaults to 50.
- `model_params.merge_alpha`: Weight for the SPECTER score when linearly mixing with Multifacet-Recommender scores. Defaults to 0.8 (recommended)
- `model_params.work_dir`: Directory where the intermediate files are stored.
- `model_params.use_cuda`: Boolean to indicate whether to use GPU (`true`) or CPU (`false`) when running SPECTER and Multifacet-Recommender. Currently, only 1 GPU is supported, but there does not seem to be necessary to have more.

Here is an example:
```
{
    "name": "iclr2020_specter",
    "dataset": {
        "directory": "./"
    },
    "model": "specter",
    "model_params": {
        "specter_dir": "../specter/",
        "average_score": false,
        "max_score": true,
        "specter_batch_size": 16,
        "publications_path": "./",
        "submissions_path": "./",
        "mfr_feature_vocab_file": "../mfr_data/feature_vocab_file",
        "mfr_checkpoint_dir": "../mfr_data/mfr_model_checkpoint/",
        "mfr_epochs": 100,
        "mfr_batch_size": 50,
        "merge_alpha": 0.8,
        "work_dir": "./",
        "use_cuda": true,
        "scores_path": "./"
    }
}
```

#### ELMo specific parameters (duplicate detection):
- `model_params.other_submissions_path`: When running ELMo, this is where the embedded abstracts/titles of the other Submissions are stored.
All the other parameters are the same as in the affinity scores.

Here is an example:
```
{
    "name": "duplicate_detection",
    "dataset": {
        "directory": "./"
    },
    "model_params": {
        "scores_path": "./",
        "use_title": false,
        "use_abstract": true,
        "use_cuda": true,
        "batch_size": 4,
        "knn": 10,
        "skip_elmo": false,
        "submissions_path": "./",
        "other_submissions_path": "./"
    }
}
```

#### TF-IDF Sparse Vector Similarity specific parameters with suggested values:
- `min_count_for_vocab`: 1,
- `random_seed`: 9,
- `max_num_keyphrases`: 25,
- `do_lower_case`: true,
- `experiment_dir`: "./"

Here is an example:

```
{
    "name": "iclr2020_reviewers_tfidf",
    "match_group": ["ICLR.cc/2020/Conference/Reviewers", "ICLR.cc/2020/Conference/Area_Chairs"],
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

In case BM25 or ELMo is used, then the Submissions (and Other Submissions when doing duplicate detection) can have the following format

```
dataset-name/
	submissions.jsonl
    other_submissions.jsonl     # Only needed for duplicate detection
```
The files `submissions.jsonl` and `other_submissions.jsonl` will have a stringified JSON submission per line.


The `archives` folder will contain the user ids of people that will review papers. The reviewers should have publications for the affinity scores to be calculated. For example, the `~User_Id1.jsonl` file will contain all the his publications.

The `submissions` folder contains all the submissions of a particular venue. The name of the file is the id used to identify the submission in OpenReview. Each file will only contain one line with all the submission content.

Alternatively, instead of using the `submissions` folder for BM25 and ELMo, the `submissions.jsonl` and `other_submissions.jsonl` will have a strigified JSON submission per line.

The stringified JSONs representing a Submission or Publication should have the following schema to work:
```
{
    id: <unique-id>,
    content: {
        title: <some-title>,
        abstract: <some-abstract>
    }
}
```
Other fields are allowed, but this is what the code will be looking for.

The `bids` folder is usually not necessary to compute affinity scores or for duplicate detection. Bids are used by the reviewers in OpenReview to select papers that they would or would not like to review. These bids are then used to compute a final affinity score to be more fair with the reviewers.

Some datasets differ slightly in terms of the format of the data; these should be accounted for in the experiment's configuration.

Some older conferences use a bidding format that differs from the default "Very High" to "Very Low" scale. This can be parameterized in the `config.json` file (e.g.) as follows:

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

## Test
The testing methodology used for the model tries to check how good the model is. We are aware that this may not be the best strategy, but it has given good results so far. The test consists on using the publications of several reviewers and take one of those publications out from the corpus. We then use that extracted publication to calculate affinity scores against the remaining publications in the corpus. If the model is good then, we expect the authors of the extracted publication to have the highest affinity scores.

This method has two obvious disadvantages:
- It only works if the author has at least two publications.
- It also assumes that all the publications of an author (or at least two of them) are very similar.

So far, we have seen that the last assumption seems to be true. We tested this on ~50,000 publications. Here are some results:

|First | ELMo | BM25 |
| ---- | ---- |----- |
| 1    |0.383 |0.318 |
| 5    |0.485 |0.486 |
| 10   |0.516 |0.538 |
| 100  |0.671 |0.735 |

This table shows that 38.3% of the time ELMo gets the author of the paper as the best ranked. Likewise, 31.8% of the time BM25 gets the author of the paper as the best ranked. We will conduct more tests in the future.
