# Open Review Expertise Evaluator
## Setting Up The Data
### Datasets
- [Gold Standard Dataset for Reviewer Paper Matching by Stelmakh et al.](https://github.com/niharshah/goldstandard-reviewer-paper-match)

Clone this repository into the following directory: `openreview-expertise/expertise/evaluation`

## Running The Evaluator
Run the following command, similar to how you would run the expertise model: `python -m expertise.evaluaton /path/to/openreview-expertise`

Tentatively, you must specify the evaluation configuration JSON directly in the `expertise/evaluation/__main__.py` file.

## Understanding The Configuration JSON

### Example Config
```
config = {
    'algo': 'specter',
    'dataset': os.path.join(args.expertise_dir, 'expertise/evaluation/goldstandard-reviewer-paper-match-main/data'),
    'predictions': os.path.join(args.expertise_dir, 'expertise/evaluation/goldstandard-reviewer-paper-match-main/predictions'),
    'destination': os.path.join(args.expertise_dir, 'expertise/evaluation/goldstandard/transformed'),
    'hist_len': 20,
    'folds': 5
}

or_config = {
    'datasets': ['goldstandard'],
    'configs': {
        'goldstandard': config
    },
    'models': ['specter2', 'scincl'],
    'evaluate': {
        'goldstandard': {
            'type': 'cross_validate',
            'settings': {
                'folds': 5,
                'samples': 10,
                'num_hyperparameter_samples': 400,
                'results_file': os.path.join(args.expertise_dir, 'expertise/evaluation/results.json')
            },
            'objective': {
                'type': 'maximize',
                'function': lambda x: (x['metrics']['easy']['point'] + x['metrics']['hard']['point'])
            },
            'hyperparameters': {
                'merge': {
                    'min': 0,
                    'max': 1
                },
                'select': {
                    'type': 'avg',
                    'min': 1,
                    'max': 10
                }
            }
        }
    },
    'use_cuda': True,
    'skip': {
        'dataset': {
            'goldstandard': True
        },
        'embedding': {
            'specter2': True,
            'scincl': True
        }
    },
}
```

### Explanation Of Fields
The `or_config` is what is used to determine the behavior of the evaluator

- `datasets`: A list of datasets used in the evaluation and it must match the implementation in the `evaluation` folder
- `configs`: Each dataset, when setting up their data, may need some user-provided settings in a JSON so each dataset in `datasets` should have a corresponding `config` JSON in `configs`
- `models`: A list of model names that are implemented in this repository that will be used to embed the text
- `evaluate`: Similar to `configs`, this JSON contains a JSON per dataset. These settings will be used to control how the evaluator encapsulates the dataset
- `evaluate.goldstandard.type`: The kind of evaluation being run on the `goldstandard` dataset
- `evaluate.goldstandard.settings`: The settings used corresponding to the algorithm selected in `evaluate.goldstandard.type`. `folds` are the number of folds used in the K-Fold cross validation.
`samples` is the number of time each reviewer's publication history is sampled, in order to average over tiebreakers for publications with the same year of publication. `num_hyperparameter_samples`
is the number of hyperparameters to sample per fold.
- `evaluate.goldstandard.objective`: The metric to optimize (maximize or minimize) for. This can be a function of the metrics stored for each hyperparameter.
- `evaluate.goldstandard.hyperparameters`: Variables to determine the distributions to sample from for each hyperparameter
- `use_cuda`: Whether or not to use CUDA compute when running the expertise models
- `skip`: Whether or not to skip generating a new dataset, and whether or not to generate new embeddings for the dataset
