"""
Benchmarking package for OpenReview expertise models.

This package contains tools for evaluating OpenReview expertise models against
standardized benchmarks, particularly the gold standard dataset from 
Stelmakh et al. (2023).
"""

from .goldstandard_evaluator import (
    AffinityModelEvaluator,
    OpenReviewModelEvaluator,
    ExternalModelEvaluator
)