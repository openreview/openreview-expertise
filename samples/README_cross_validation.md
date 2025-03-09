# Cross-Validation for Hyperparameter Optimization

This document explains how to use the cross-validation functionality for hyperparameter optimization in the OpenReview expertise benchmark system.

## Overview

The cross-validation implementation uses a nested cross-validation approach:

1. **Outer Loop**: Each of the 10 existing dataset iterations serves once as the test set.
2. **Inner Loop**: The remaining 9 iterations are split into training and validation sets using k-fold cross-validation (default k=3).

This approach allows for robust hyperparameter selection while avoiding data leakage and providing reliable generalization estimates.

## Usage

### Command Line

```bash
python -m expertise.benchmark.goldstandard_evaluator \
  --mode cv \
  --goldstandard_dir /path/to/goldstandard \
  --output_dir /path/to/output \
  --model_name my_model_name \
  --model_type specter \
  --hyperparameter_space /path/to/hyperparameter_space.json \
  --inner_cv_splits 3 \
  --random_state 42
```

### Parameters

- `--mode cv`: Specifies that we want to use cross-validation mode
- `--goldstandard_dir`: Path to the gold standard dataset directory
- `--output_dir`: Directory to save evaluation results
- `--model_name`: Name of the model being evaluated
- `--model_type`: Type of model (e.g., 'specter', 'bm25', 'specter+mfr')
- `--hyperparameter_space`: Path to JSON file defining hyperparameter space
- `--inner_cv_splits`: Number of inner CV splits (default: 3)
- `--random_state`: Random seed for cross-validation (default: 42)

### Hyperparameter Space Definition

The hyperparameter space is defined in a JSON file that specifies the parameters to tune and their possible values. For example:

```json
{
    "max_score": [true, false],
    "average_score": [true, false],
    "batch_size": [8, 16, 32]
}
```

Sample hyperparameter space files are provided:
- `samples/hyperparameter_space_specter.json`
- `samples/hyperparameter_space_bm25.json`

## Output

The cross-validation process produces the following outputs in the `output_dir/hyperparameter_tuning` directory:

1. **Inner CV Results**: For each outer fold, detailed results from the inner cross-validation are saved in `inner_cv_results_fold_N.json`.
2. **Final CV Results**: The overall cross-validation results are saved in `model_name_cv_tuning_results.json`.
3. **Best Hyperparameters**: The final selected hyperparameters are saved in `model_name_best_hyperparams.json`.

The final results include:
- Generalization score estimated from the outer test folds
- Standard deviation of the generalization score
- Best hyperparameters selected (most frequently chosen across outer folds)
- Detailed results for each outer fold

## Implementation Details

The cross-validation is implemented in the `CrossValidatedModelEvaluator` class, which extends the `OpenReviewModelEvaluator`.

The class handles:
1. Generation of all hyperparameter combinations to evaluate
2. Creation of inner cross-validation folds
3. Training and evaluation of the model with each hyperparameter combination
4. Selection of the best hyperparameters for each outer fold
5. Final evaluation on the outer test sets
6. Aggregation of results and selection of final hyperparameters

## Example

To run cross-validation for the SPECTER model:

```bash
python -m expertise.benchmark.goldstandard_evaluator \
  --mode cv \
  --goldstandard_dir /path/to/goldstandard \
  --output_dir /path/to/results \
  --model_name specter_tuned \
  --model_type specter \
  --hyperparameter_space samples/hyperparameter_space_specter.json
``` 