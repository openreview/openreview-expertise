# OpenReview Expertise Benchmarks

This directory contains benchmarking tools for evaluating OpenReview affinity models against standard datasets.

## Gold Standard Evaluator

The Gold Standard Evaluator (`goldstandard_evaluator.py`) allows you to evaluate models against the gold standard dataset for reviewer-paper matching from [Stelmakh et al. (2023)](https://arxiv.org/abs/2303.16750).

### Usage

You can use the evaluator in two ways:

#### 1. Command-line interface

```bash
# Evaluate an OpenReview model
python -m expertise.benchmark.goldstandard_evaluator \
    --goldstandard_dir /path/to/goldstandard-dataset \
    --output_dir ./results \
    --model_name "my_bm25_model" \
    --mode openreview \
    --model_type bm25

# Evaluate external model predictions
python -m expertise.benchmark.goldstandard_evaluator \
    --goldstandard_dir /path/to/goldstandard-dataset \
    --output_dir ./results \
    --model_name "my_custom_model" \
    --mode external \
    --predictions_dir ./my_model_predictions
```

#### 2. Programmatic usage

```python
from expertise.benchmark import OpenReviewModelEvaluator

# Create the evaluator
evaluator = OpenReviewModelEvaluator(
    goldstandard_dir="/path/to/goldstandard-dataset",
    output_dir="./results",
    model_name="specter_test", 
    model_type="specter"
)

# Run evaluation
results = evaluator.run_evaluation()

# Print results
print(f"Loss: {results['summary']['loss']}")
print(f"Easy Triples: {results['summary']['easy_triples']}")
print(f"Hard Triples: {results['summary']['hard_triples']}")
```

### Supported Models

The evaluator supports all models available in the OpenReview expertise repository:

- `bm25`: Classic BM25 text similarity model
- `specter`: SPECTER embedding model for scientific papers
- `specter+mfr`: Ensemble of SPECTER and Multi-facet Recommender
- `specter2+scincl`: Ensemble of SPECTER2 and SciNCL

### Evaluation Metrics

1. **Loss**: Weighted Kendall's Tau (lower is better)
2. **Easy Triples**: Accuracy on paper pairs with clearly different expertise levels
3. **Hard Triples**: Accuracy on paper pairs with similar high expertise levels

## Adding New Benchmarks

To add a new benchmark, create a new module in this directory and update the `__init__.py` to expose the relevant classes.