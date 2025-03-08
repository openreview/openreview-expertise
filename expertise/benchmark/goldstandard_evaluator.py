"""
Evaluator interface for OpenReview expertise models using the gold standard dataset.

This module provides a standardized way to evaluate affinity scoring models from 
the OpenReview expertise repository against the gold standard dataset developed by 
Stelmakh et al. (2023). It handles loading the gold standard dataset, running models 
from openreview-expertise, and computing evaluation metrics.

This evaluator is integrated directly with the openreview-expertise repository
and can be run to benchmark different models against the gold standard dataset.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
import sys
import importlib.util
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple

# Try to import torch for CUDA availability check, but don't require it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AffinityModelEvaluator:
    """
    Base class for evaluating affinity scoring models against the gold standard dataset.
    
    This class provides common functionality for evaluating different types of affinity
    scoring models. It handles loading the gold standard dataset, preparing model
    predictions in the required format, and computing evaluation metrics.
    
    The evaluator implements the methodology from Stelmakh et al. (2023), computing:
    1. Main metric: Weighted Kendall's Tau - measuring the model's ability to correctly 
       rank papers according to expertise level
    2. Easy Triples: Accuracy on paper pairs with clear expertise differences
    3. Hard Triples: Accuracy on paper pairs with similar high expertise levels
    
    All evaluators should inherit from this class and implement the `evaluate_model` method.
    """
    
    def __init__(
        self,
        goldstandard_dir: str,
        output_dir: str,
        model_name: str,
    ):
        """
        Initialize the evaluator.
        
        Args:
            goldstandard_dir: Path to the gold standard dataset directory
            output_dir: Directory to save evaluation results
            model_name: Name of the model being evaluated
        """
        self.goldstandard_dir = Path(goldstandard_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the gold standard references
        self.references = self._load_gold_standard_references()
        
        # Extract reviewers and papers
        self.all_reviewers = list(self.references.keys())
        self.all_papers = set()
        for rev in self.references:
            self.all_papers = self.all_papers.union(self.references[rev].keys())
            
        # Prepare bootstrap samples for confidence intervals
        self.bootstraps = [np.random.choice(self.all_reviewers, len(self.all_reviewers), replace=True) for _ in range(1000)]
    
    def _load_gold_standard_references(self) -> Dict:
        """
        Load the gold standard dataset references.
        
        Returns:
            Dictionary mapping reviewers to papers with expertise scores
        """
        evaluations_path = self.goldstandard_dir / "data" / "evaluations.csv"
        if not evaluations_path.exists():
            raise FileNotFoundError(f"Gold standard dataset not found at {evaluations_path}")
        
        df = pd.read_csv(evaluations_path, sep='\t')
        return self._convert_dataframe_to_dicts(df)
    
    def _convert_dataframe_to_dicts(self, df: pd.DataFrame) -> Dict:
        """
        Convert the gold standard dataframe to dictionaries for evaluation.
        
        Args:
            df: Dataframe containing gold standard evaluations
            
        Returns:
            Dictionary mapping reviewers to papers with expertise scores
        """
        # Try to use the original helper function if available
        try:
            # Import helper from goldstandard repository
            import sys
            sys.path.append(str(self.goldstandard_dir / "scripts"))
            import helpers
            references, _ = helpers.to_dicts(df)
            return references
        except (ImportError, AttributeError):
            # Fallback to our own implementation
            logger.warning("Could not import helpers from goldstandard repository. Using fallback implementation.")
            references = {}
            
            for _, row in df.iterrows():
                reviewer_id = row['ParticipantID']
                references[reviewer_id] = {}
                
                for i in range(1, 11):  # 10 papers per reviewer
                    paper_col = f'Paper{i}'
                    expertise_col = f'Expertise{i}'
                    
                    if pd.notna(row[paper_col]) and pd.notna(row[expertise_col]):
                        paper_id = row[paper_col]
                        expertise = row[expertise_col]
                        references[reviewer_id][paper_id] = expertise
            
            return references
    
    def compute_main_metric(
        self, 
        predictions: Dict, 
        valid_papers: set, 
        valid_reviewers: List
    ) -> float:
        """
        Compute the main evaluation metric (weighted kendall's tau).
        
        This is the primary metric used in the gold standard paper. It measures how well 
        the model ranks pairs of papers according to the ground truth expertise ratings.
        Errors are weighted by the absolute difference in expertise ratings.
        
        Args:
            predictions: Dictionary of predicted affinities
            valid_papers: Set of papers to include in evaluation
            valid_reviewers: List of reviewers to include in evaluation
            
        Returns:
            Loss value (lower is better, 0 is perfect ranking)
        """
        from itertools import combinations
        
        max_loss, loss = 0, 0

        for reviewer in valid_reviewers:
            # Skip reviewers not in the reference dataset
            if reviewer not in self.references:
                continue
                
            papers = list(self.references[reviewer].keys())

            for p1, p2 in combinations(papers, 2):
                # Skip papers not in the valid set
                if p1 not in valid_papers or p2 not in valid_papers:
                    continue

                # Calculate prediction and ground truth differences
                pred_diff = predictions[reviewer][p1] - predictions[reviewer][p2]
                true_diff = self.references[reviewer][p1] - self.references[reviewer][p2]

                # Update maximum possible loss
                max_loss += np.abs(true_diff)

                # Handle ties in predictions (half penalty)
                if pred_diff * true_diff == 0:  # One is zero or they have opposite signs
                    loss += np.abs(true_diff) / 2

                # Handle ranking inversions (full penalty)
                if pred_diff * true_diff < 0:  # Opposite signs means incorrect ranking
                    loss += np.abs(true_diff)

        # Return normalized loss or 0 if no valid pairs were found
        return loss / max_loss if max_loss > 0 else 0
    
    def compute_resolution(
        self, 
        predictions: Dict, 
        valid_papers: set, 
        valid_reviewers: List, 
        regime: str = 'easy'
    ) -> Dict:
        """
        Compute resolution ability of the model for easy/hard pairs of papers.
        
        This metric measures the model's ability to distinguish between papers of different
        expertise levels:
        - 'easy': One paper has high expertise (4+), one has low expertise (2-) 
        - 'hard': Both papers have high expertise (4+), but differ in rating
        
        Args:
            predictions: Dictionary of predicted affinities
            valid_papers: Set of papers to include in evaluation
            valid_reviewers: List of reviewers to include in evaluation
            regime: 'easy' or 'hard' triples evaluation
            
        Returns:
            Dictionary with resolution score, correct count, and total count
        """
        from itertools import combinations
        
        if regime not in {'easy', 'hard'}:
            raise ValueError("Regime must be either 'easy' or 'hard'")

        num_pairs = 0
        num_correct = 0

        for reviewer in valid_reviewers:
            # Skip reviewers not in the reference dataset
            if reviewer not in self.references:
                continue
                
            papers = list(self.references[reviewer].keys())

            for p1, p2 in combinations(papers, 2):
                # Skip papers not in the valid set
                if p1 not in valid_papers or p2 not in valid_papers:
                    continue

                s1 = self.references[reviewer][p1]
                s2 = self.references[reviewer][p2]

                # We only look at pairs of papers that are not tied in terms of the expertise
                if s1 == s2:
                    continue

                # Hard-coded parameters from gold standard paper
                # Hard regime: Both papers have high expertise (4+), but differ in rating
                if regime == 'hard' and min(s1, s2) < 4:
                    continue

                # Easy regime: One paper has high expertise (4+), one has low expertise (2-)
                if regime == 'easy' and (max(s1, s2) < 4 or min(s1, s2) > 2):
                    continue

                # Count this pair for evaluation
                num_pairs += 1
                
                # Calculate differences
                pred_diff = predictions[reviewer][p1] - predictions[reviewer][p2]
                true_diff = s1 - s2

                # An algorithm is correct if the ordering of predicted similarities agrees
                # with the ordering of the ground-truth expertise (same sign)
                if pred_diff * true_diff > 0:
                    num_correct += 1

        # Calculate accuracy or return 0 if no pairs were found
        accuracy = num_correct / num_pairs if num_pairs > 0 else 0
        
        return {
            'score': accuracy, 
            'correct': num_correct, 
            'total': num_pairs
        }
    
    def evaluate_model(self, config_path: str) -> Dict:
        """
        Evaluate a model using the provided configuration.
        
        This method should be implemented by each specific evaluator.
        
        Args:
            config_path: Path to the model configuration file
            
        Returns:
            Dictionary of evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate_model")
    
    def run_evaluation(self, config_path: str) -> Dict:
        """
        Run the evaluation and return results.
        
        Args:
            config_path: Path to the model configuration file
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Starting evaluation of {self.model_name}")
        results = self.evaluate_model(config_path)
        
        # Save results
        results_path = self.output_dir / f"{self.model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Evaluation complete. Results saved to {results_path}")
        return results


class OpenReviewModelEvaluator(AffinityModelEvaluator):
    """
    Evaluator for models from the OpenReview expertise repository.
    
    This class runs OpenReview expertise models on the gold standard dataset
    and computes evaluation metrics.
    """
    
    def __init__(
        self,
        goldstandard_dir: str,
        output_dir: str,
        model_name: str,
        model_type: str,
    ):
        """
        Initialize the OpenReview model evaluator.
        
        Args:
            goldstandard_dir: Path to the gold standard dataset directory
            output_dir: Directory to save evaluation results
            model_name: Name of the model being evaluated
            model_type: Type of model (e.g., 'specter', 'bm25', 'specter+mfr')
        """
        super().__init__(goldstandard_dir, output_dir, model_name)
        self.model_type = model_type
        
        # Path to the gold standard evaluation datasets
        self.evaluation_datasets_dir = self.goldstandard_dir / "evaluation_datasets"
        if not self.evaluation_datasets_dir.exists():
            raise FileNotFoundError(
                f"Evaluation datasets directory not found at {self.evaluation_datasets_dir}. "
                f"Please ensure the goldstandard repository structure is correct."
            )
        
        # Path to store predictions
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset_config(self, dataset_path: str) -> Dict:
        """
        Prepare a configuration for dataset loading.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Configuration dictionary for dataset loading
        """
        return {
            "name": f"{self.model_name}",
            "dataset": {
                "directory": str(dataset_path)
            }
        }
    
    def prepare_model_config(self, dataset_config: Dict) -> Dict:
        """
        Prepare a configuration for the model.
        
        Args:
            dataset_config: Dataset configuration dictionary
            
        Returns:
            Configuration dictionary for the model
        """
        config = dataset_config.copy()
        config["model"] = self.model_type
        
        # Since we're in the same environment as openreview-expertise, we can use
        # the default paths that would be set up by the installation process
        
        # Determine CUDA availability
        use_cuda = False
        if TORCH_AVAILABLE:
            try:
                use_cuda = torch.cuda.is_available()
                logger.info(f"CUDA availability: {use_cuda}")
            except Exception as e:
                logger.warning(f"Error checking CUDA availability: {e}")
                
        # Add model-specific parameters based on the model type
        if self.model_type == "bm25":
            config["model_params"] = {
                "scores_path": str(self.predictions_dir),
                "use_title": True,
                "use_abstract": True,
                "workers": 4,
                "publications_path": str(self.predictions_dir),
                "submissions_path": str(self.predictions_dir)
            }
        elif self.model_type == "specter":
            config["model_params"] = {
                # Use the environment variable if set, otherwise let openreview-expertise use its default
                "specter_dir": os.environ.get("SPECTER_DIR", ""),
                "work_dir": str(self.predictions_dir),
                "average_score": False,
                "max_score": True,
                "use_cuda": use_cuda,
                "batch_size": 16,
                "publications_path": str(self.predictions_dir),
                "submissions_path": str(self.predictions_dir),
                "scores_path": str(self.predictions_dir)
            }
        elif self.model_type == "specter+mfr":
            config["model_params"] = {
                "specter_dir": os.environ.get("SPECTER_DIR", ""),
                "average_score": False,
                "max_score": True,
                "specter_batch_size": 16,
                "publications_path": str(self.predictions_dir),
                "submissions_path": str(self.predictions_dir),
                "mfr_feature_vocab_file": os.environ.get("MFR_VOCAB_FILE", ""),
                "mfr_checkpoint_dir": os.environ.get("MFR_CHECKPOINT_DIR", ""),
                "mfr_epochs": 100,
                "mfr_batch_size": 50,
                "merge_alpha": 0.8,
                "work_dir": str(self.predictions_dir),
                "use_cuda": use_cuda,
                "scores_path": str(self.predictions_dir)
            }
        elif self.model_type == "specter2+scincl":
            config["model_params"] = {
                "specter_dir": os.environ.get("SPECTER_DIR", ""),
                "work_dir": str(self.predictions_dir),
                "average_score": False,
                "max_score": True,
                "batch_size": 16,
                "use_cuda": use_cuda,
                "publications_path": str(self.predictions_dir),
                "submissions_path": str(self.predictions_dir),
                "scores_path": str(self.predictions_dir)
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Remove empty values since we want to use the defaults from openreview-expertise
        for key, value in list(config["model_params"].items()):
            if value == "":
                del config["model_params"][key]
                logger.info(f"Using default value for {key}")
                
        return config
    
    def run_model(self, dataset_path: str) -> str:
        """
        Run the model on a dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Path to the output predictions file
        """
        # Check that the dataset exists and is in the expected format
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        # Check for archives and submissions directories (required by OpenReview)
        archives_path = dataset_path / "archives"
        submissions_path = dataset_path / "submissions"
        
        if not archives_path.exists() or not submissions_path.exists():
            logger.warning(f"Dataset at {dataset_path} may not be in the expected format for OpenReview. "
                          f"Expected to find 'archives' and 'submissions' directories.")
        
        # Prepare configurations
        dataset_config = self.prepare_dataset_config(str(dataset_path))
        model_config = self.prepare_model_config(dataset_config)
        
        # Write config to file
        config_path = self.predictions_dir / f"{self.model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Import execute_expertise - should be available directly now that we're inside the package
        from expertise.execute_expertise import execute_expertise
        
        # Execute the model with careful error handling
        try:
            logger.info(f"Executing {self.model_type} model on dataset {dataset_path}")
            execute_expertise(model_config)
            
            # Return path to the scores file
            scores_file_path = self.predictions_dir / f"{self.model_name}.csv"
            
            # Check that the output file was created
            if not scores_file_path.exists():
                raise FileNotFoundError(
                    f"Model execution completed, but no scores file was created at {scores_file_path}. "
                    f"Check the model logs for errors."
                )
                
            return str(scores_file_path)
            
        except Exception as e:
            logger.error(f"Error executing model: {e}")
            
            # Try to find any partial output that might be available
            scores_file_path = self.predictions_dir / f"{self.model_name}.csv"
            if scores_file_path.exists():
                logger.info(f"Found partial output at {scores_file_path}, will attempt to continue with evaluation")
                return str(scores_file_path)
            
            raise RuntimeError(f"Failed to execute model: {e}")

    def process_predictions(self, scores_file: str) -> Dict:
        """
        Process the model predictions into the required format.
        
        Args:
            scores_file: Path to the scores file produced by the model
            
        Returns:
            Dictionary mapping reviewers to papers with affinity scores
        """

        def _preprocess_scores_file():
            # Helper function to swap columns 0 and 1 and add headers
            # Writes to a new file and returns the path
            new_file = scores_file.replace(".csv", "_preprocessed.csv")
            with open(scores_file, 'r') as f, open(new_file, 'w') as out:
                lines = f.readlines()
                out.write("reviewer,submission,score\n")
                for line in lines:
                    parts = line.strip().split(",")
                    out.write(f"{parts[1][1:]},{parts[0]},{parts[2]}\n") ## Swap columns and chop tilde
            return new_file

        try:
            # Preprocess CSV
            preprocessed_file = _preprocess_scores_file()

            # Read scores
            scores_df = pd.read_csv(preprocessed_file)
            
            # Validate the file has the required columns
            required_columns = ['reviewer', 'submission', 'score']
            missing_columns = [col for col in required_columns if col not in scores_df.columns]
            
            if missing_columns:
                raise ValueError(
                    f"Scores file is missing required columns: {missing_columns}. "
                    f"Expected columns: {required_columns}, found: {scores_df.columns.tolist()}"
                )
            
            # Convert to dictionary format
            predictions = {}
            for _, row in scores_df.iterrows():
                try:
                    reviewer_id = str(row['reviewer'])
                    paper_id = str(row['submission'])
                    score = float(row['score'])
                    
                    if reviewer_id not in predictions:
                        predictions[reviewer_id] = {}
                        
                    predictions[reviewer_id][paper_id] = score
                    
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error processing row {row}: {e}. Skipping.")
                    continue
                    
            if not predictions:
                raise ValueError("No valid predictions found in the scores file.")
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing predictions file {preprocessed_file}: {e}")
            raise
    
    def convert_to_gold_standard_format(self, predictions: Dict) -> Dict:
        """
        Convert predictions to match the gold standard reference format.
        
        This method ensures that all reviewer-paper pairs in the gold standard dataset
        have corresponding prediction values, filling in missing values with defaults.
        
        Args:
            predictions: Dictionary of predicted affinities
            
        Returns:
            Predictions reformatted to match gold standard references
        """
        if not isinstance(predictions, dict):
            raise TypeError(f"Predictions must be a dictionary, got {type(predictions)}")
            
        formatted_predictions = {}
        missing_reviewers = []
        missing_predictions_count = 0
        
        # Ensure all reviewers and papers from gold standard are included
        for reviewer in self.all_reviewers:
            if reviewer not in formatted_predictions:
                formatted_predictions[reviewer] = {}
                
            # Check if reviewer is missing from predictions
            reviewer_missing = reviewer not in predictions
            if reviewer_missing:
                missing_reviewers.append(reviewer)
                
            for paper in self.all_papers:
                # If we have a prediction for this paper, use it
                if not reviewer_missing and paper in predictions[reviewer]:
                    formatted_predictions[reviewer][paper] = predictions[reviewer][paper]
                # Otherwise assign a default value (0)
                else:
                    formatted_predictions[reviewer][paper] = 0
                    missing_predictions_count += 1
        
        # Log information about missing predictions
        if missing_reviewers:
            logger.warning(f"{len(missing_reviewers)} reviewers from gold standard dataset missing in predictions")
            
        if missing_predictions_count > 0:
            total_pairs = len(self.all_reviewers) * len(self.all_papers)
            logger.warning(f"Filled in {missing_predictions_count}/{total_pairs} missing predictions with default value 0")
            
        return formatted_predictions
    
    def evaluate_model(self, config_path: Optional[str] = None) -> Dict:
        """
        Evaluate the model on the gold standard dataset.
        
        This method runs the evaluation across all 10 iteration datasets as defined in the
        gold standard methodology paper. It handles computing metrics and aggregating results.
        
        Args:
            config_path: Path to the model configuration file (not used here)
            
        Returns:
            Dictionary of evaluation results
        """
        results = {'pointwise': [], 'variations': [], 'easy_triples': [], 'hard_triples': []}
        completed_iterations = []
        
        # Evaluate on 10 datasets as per gold standard methodology
        for iteration in range(1, 11):
            dataset_path = self.evaluation_datasets_dir / f"d_20_{iteration}"
            
            # Validate dataset existence
            if not dataset_path.exists():
                logger.error(f"Dataset not found at {dataset_path}. Skipping iteration {iteration}.")
                continue
                
            logger.info(f"Evaluating on dataset {dataset_path} (iteration {iteration}/10)")
            
            try:
                # Step 1: Run model to get predictions
                try:
                    scores_file = self.run_model(dataset_path)
                except Exception as e:
                    logger.error(f"Error running model for iteration {iteration}: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Step 2: Process predictions
                try:
                    raw_predictions = self.process_predictions(scores_file)
                except Exception as e:
                    logger.error(f"Error processing predictions for iteration {iteration}: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Step 3: Convert to gold standard format
                try:
                    predictions = self.convert_to_gold_standard_format(raw_predictions)
                    
                    # Save predictions in gold standard format
                    pred_output_path = self.predictions_dir / f"{self.model_name}_d_20_{iteration}_ta.json"
                    with open(pred_output_path, 'w') as f:
                        json.dump(predictions, f)
                except Exception as e:
                    logger.error(f"Error converting predictions for iteration {iteration}: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Step 4: Compute all metrics
                try:
                    # Main metric
                    score = self.compute_main_metric(predictions, self.all_papers, self.all_reviewers)
                    
                    # Bootstrap variations for confidence intervals
                    variations = [self.compute_main_metric(predictions, self.all_papers, bootstrap) 
                                for bootstrap in self.bootstraps]
                    
                    # Resolution metrics
                    easy_triples = self.compute_resolution(
                        predictions, self.all_papers, self.all_reviewers, regime='easy'
                    )
                    hard_triples = self.compute_resolution(
                        predictions, self.all_papers, self.all_reviewers, regime='hard'
                    )
                    
                    # Store all results
                    results['pointwise'].append(score)
                    results['variations'].append(variations)
                    results['easy_triples'].append(easy_triples)
                    results['hard_triples'].append(hard_triples)
                    completed_iterations.append(iteration)
                    
                    logger.info(
                        f"Iteration {iteration} complete. Loss: {score:.4f}, "
                        f"Easy: {easy_triples['score']:.4f}, Hard: {hard_triples['score']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error computing metrics for iteration {iteration}: {e}")
                    logger.error(traceback.format_exc())
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error in iteration {iteration}: {e}")
                logger.error(traceback.format_exc())
                # Continue with next iteration instead of failing completely
                continue
            
        # Check if we have enough data to compute summary statistics
        if not results['pointwise']:
            logger.error("No evaluation results were collected. Cannot compute summary statistics.")
            summary = {
                'model': self.model_name,
                'model_type': self.model_type,
                'error': "No valid evaluation results were collected"
            }
        else:
            # Calculate summary statistics
            summary = {
                'model': self.model_name,
                'model_type': self.model_type,
                'iterations_completed': len(results['pointwise']),
                'completed_iterations': completed_iterations,
                'loss': round(np.mean(results['pointwise']), 2),
                'easy_triples': round(np.mean([r['score'] for r in results['easy_triples']]), 2),
                'hard_triples': round(np.mean([r['score'] for r in results['hard_triples']]), 2),
            }
            
            # Get 95% confidence interval if we have bootstrap data
            if results['variations']:
                try:
                    boot = np.matrix(results['variations']).mean(axis=0).tolist()[0]
                    ci_low = round(np.percentile(boot, 2.5), 2)
                    ci_high = round(np.percentile(boot, 97.5), 2)
                    summary['confidence_interval'] = [ci_low, ci_high]
                except Exception as e:
                    logger.warning(f"Error computing confidence intervals: {e}")
                    summary['confidence_interval'] = None
        
        results['summary'] = summary
        
        # Log the summary
        logger.info(f"Evaluation complete. Summary: {json.dumps(summary, indent=2)}")
        
        return results


class ExternalModelEvaluator(AffinityModelEvaluator):
    """
    Evaluator for external models not from the OpenReview expertise repository.
    
    This class processes pre-computed predictions from external models and
    computes evaluation metrics. It expects predictions to be saved as JSON files
    following the gold standard format.
    
    This evaluator is useful when:
    1. You have developed your own affinity scoring model outside of openreview-expertise
    2. You want to evaluate a model that can't easily run in the same environment
    3. You've pre-computed predictions to save time during evaluation
    """
    
    def __init__(
        self,
        goldstandard_dir: str,
        output_dir: str,
        model_name: str,
    ):
        """
        Initialize the external model evaluator.
        
        Args:
            goldstandard_dir: Path to the gold standard dataset directory
            output_dir: Directory to save evaluation results
            model_name: Name of the model being evaluated
        """
        super().__init__(goldstandard_dir, output_dir, model_name)
    
    def evaluate_model(self, predictions_dir: str) -> Dict:
        """
        Evaluate the model based on pre-computed predictions.
        
        Args:
            predictions_dir: Directory containing the model's predictions
            
        Returns:
            Dictionary of evaluation results
        """
        predictions_dir = Path(predictions_dir)
        if not predictions_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found at {predictions_dir}")
            
        results = {'pointwise': [], 'variations': [], 'easy_triples': [], 'hard_triples': []}
        missing_files = []
        
        # Check that prediction files are available
        for iteration in range(1, 11):
            f_name = f"{self.model_name}_d_20_{iteration}_ta.json"
            if not (predictions_dir / f_name).exists():
                missing_files.append(f_name)
                
        if missing_files:
            logger.warning(f"Some prediction files are missing: {missing_files}")
            logger.warning("Will evaluate using only available files.")
        
        # Evaluate each prediction file
        valid_iterations = []
        for iteration in range(1, 11):
            f_name = f"{self.model_name}_d_20_{iteration}_ta.json"
            f_path = predictions_dir / f_name
            
            if not f_path.exists():
                logger.warning(f"Skipping missing file: {f_name}")
                continue
                
            try:
                logger.info(f"Evaluating predictions from {f_name}")
                
                # Load predictions
                try:
                    with open(f_path, 'r') as handler:
                        predictions = json.load(handler)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON in {f_path}: {e}")
                    continue
                    
                # Validate predictions format
                if not isinstance(predictions, dict):
                    logger.error(f"Invalid predictions format in {f_path}: expected dictionary")
                    continue
                    
                # Verify that predictions contain data for at least some reviewers
                if not predictions:
                    logger.error(f"Empty predictions in {f_path}")
                    continue
                    
                # Compute metrics
                score = self.compute_main_metric(predictions, self.all_papers, self.all_reviewers)
                variations = [self.compute_main_metric(predictions, self.all_papers, bootstrap) 
                              for bootstrap in self.bootstraps]
                
                easy_triples = self.compute_resolution(predictions, self.all_papers, self.all_reviewers, regime='easy')
                hard_triples = self.compute_resolution(predictions, self.all_papers, self.all_reviewers, regime='hard')
                
                results['pointwise'].append(score)
                results['variations'].append(variations)
                results['easy_triples'].append(easy_triples)
                results['hard_triples'].append(hard_triples)
                valid_iterations.append(iteration)
                
                logger.info(f"Iteration {iteration} complete. Loss: {score:.4f}, Easy: {easy_triples['score']:.4f}, Hard: {hard_triples['score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing predictions file {f_path}: {e}")
                continue
        
        # Check if we have enough data to compute summary statistics
        if not results['pointwise']:
            logger.error("No evaluation results were collected. Cannot compute summary statistics.")
            summary = {
                'model': self.model_name,
                'error': "No valid evaluation results were collected"
            }
        else:
            # Calculate summary statistics
            summary = {
                'model': self.model_name,
                'iterations_completed': len(results['pointwise']),
                'iterations_used': valid_iterations,
                'loss': round(np.mean(results['pointwise']), 2),
                'easy_triples': round(np.mean([r['score'] for r in results['easy_triples']]), 2),
                'hard_triples': round(np.mean([r['score'] for r in results['hard_triples']]), 2),
            }
            
            # Get 95% confidence interval if we have bootstrap data
            if results['variations']:
                try:
                    boot = np.matrix(results['variations']).mean(axis=0).tolist()[0]
                    ci_low = round(np.percentile(boot, 2.5), 2)
                    ci_high = round(np.percentile(boot, 97.5), 2)
                    summary['confidence_interval'] = [ci_low, ci_high]
                except Exception as e:
                    logger.warning(f"Error computing confidence intervals: {e}")
                    summary['confidence_interval'] = None
        
        results['summary'] = summary
        
        # Log the summary
        logger.info(f"Evaluation complete. Summary: {json.dumps(summary, indent=2)}")
        
        return results


def main():
    """Main function to run the evaluator from the command line."""
    parser = argparse.ArgumentParser(description='Evaluate affinity scoring models')
    
    parser.add_argument('--goldstandard_dir', type=str, required=True,
                        help='Path to the gold standard dataset directory')
    # No longer needed since we're inside the openreview-expertise package
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model being evaluated')
    parser.add_argument('--mode', type=str, choices=['openreview', 'external'], required=True,
                        help='Evaluation mode: openreview (run model) or external (use precomputed predictions)')
    
    # Arguments specific to OpenReview mode
    parser.add_argument('--model_type', type=str, 
                        choices=['bm25', 'specter', 'specter+mfr', 'specter2+scincl'],
                        help='Type of OpenReview model (required for openreview mode)')
    parser.add_argument('--config_path', type=str,
                        help='Path to model configuration file (for openreview mode)')
    
    # Arguments specific to External mode
    parser.add_argument('--predictions_dir', type=str,
                        help='Directory containing precomputed predictions (required for external mode)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'openreview' and not args.model_type:
        parser.error("--model_type is required for openreview mode")
    elif args.mode == 'external' and not args.predictions_dir:
        parser.error("--predictions_dir is required for external mode")
    
    # Create and run the appropriate evaluator
    if args.mode == 'openreview':
        evaluator = OpenReviewModelEvaluator(
            goldstandard_dir=args.goldstandard_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            model_type=args.model_type
        )
        results = evaluator.run_evaluation(args.config_path)
    else:  # external mode
        evaluator = ExternalModelEvaluator(
            goldstandard_dir=args.goldstandard_dir,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        results = evaluator.evaluate_model(args.predictions_dir)
    
    # Print summary results
    print("\nEvaluation Summary:")
    summary = results['summary']
    print(f"Model: {summary['model']}")
    
    if 'error' in summary:
        print(f"Error: {summary['error']}")
    else:
        print(f"Iterations completed: {summary.get('iterations_completed', 'N/A')}/{10}")
        
        # Print which iterations were completed
        if 'completed_iterations' in summary:
            completed = summary['completed_iterations']
            if len(completed) < 10:
                missing = [i for i in range(1, 11) if i not in completed]
                print(f"Completed: {completed}")
                print(f"Missing: {missing}")
                
        print(f"Loss: {summary.get('loss', 'N/A')}")
        
        if 'confidence_interval' in summary:
            print(f"95% CI: {summary['confidence_interval']}")
            
        print(f"Easy Triples Accuracy: {summary.get('easy_triples', 'N/A')}")
        print(f"Hard Triples Accuracy: {summary.get('hard_triples', 'N/A')}")


if __name__ == '__main__':
    main()