import os, sys, json, argparse
import numpy as np
import pandas as pd

from expertise.evaluation import OpenReviewExpertiseEvaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('expertise_dir', help='directory to openreview-expertise')
    args = parser.parse_args()

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
                    'num_hyperparameter_samples': 350,
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
    eval = OpenReviewExpertiseEvaluation(or_config)
    eval.run()

'''or_config = {
        'datasets': ['goldstandard'],
        'configs': {
            'goldstandard': config
        },
        'models': ['specter2', 'scincl'],
        'evaluate': {
            'goldstandard': {
                'type': 'validate',
                'settings': {
                    'settings_file': os.path.join(args.expertise_dir, 'expertise/evaluation/best_settings.json'),
                    'results_file': os.path.join(args.expertise_dir, 'expertise/evaluation/results.json'),
                    'results_dir': os.path.join(args.expertise_dir, 'expertise/evaluation/results')
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
    }'''