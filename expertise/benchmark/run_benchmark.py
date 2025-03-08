#!/usr/bin/env python3
"""
Script to run benchmarks for OpenReview expertise models.

This script provides a convenient way to benchmark models against the
gold standard dataset.

Example usage:
    python run_benchmark.py --model bm25 --goldstandard /path/to/goldstandard
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from expertise.benchmark import OpenReviewModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run OpenReview Expertise benchmarks')
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25', 'specter', 'specter+mfr', 'specter2+scincl'],
                        help='Model to benchmark')
    parser.add_argument('--goldstandard', type=str, required=True,
                        help='Path to the gold standard dataset')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                        help='Directory to save benchmark results')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for the benchmark run (defaults to model name)')
    return parser.parse_args()

def main():
    """Run the benchmark."""
    args = parse_args()
    
    # Set default name if not provided
    model_name = args.name if args.name else f"{args.model}_benchmark"
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify gold standard dataset exists
    goldstandard_dir = Path(args.goldstandard)
    if not goldstandard_dir.exists():
        logger.error(f"Gold standard dataset not found at {goldstandard_dir}")
        sys.exit(1)
    
    logger.info(f"Starting benchmark of model {args.model}")
    logger.info(f"Using gold standard dataset: {goldstandard_dir}")
    logger.info(f"Saving results to: {output_dir}")
    
    # Create and run the evaluator
    try:
        evaluator = OpenReviewModelEvaluator(
            goldstandard_dir=str(goldstandard_dir),
            output_dir=str(output_dir),
            model_name=model_name,
            model_type=args.model
        )
        
        results = evaluator.run_evaluation()
        
        # Print summary
        summary = results['summary']
        print("\nBenchmark Results:")
        print(f"Model: {args.model}")
        print(f"Loss: {summary['loss']} (95% CI: {summary.get('confidence_interval', 'N/A')})")
        print(f"Easy Triples Accuracy: {summary['easy_triples']}")
        print(f"Hard Triples Accuracy: {summary['hard_triples']}")
        
        logger.info(f"Benchmark complete! Results saved to {output_dir}/{model_name}_results.json")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()