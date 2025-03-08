"""
Tests for the command-line interface of the benchmark goldstandard_evaluator module.

These tests validate the CLI functionality, including argument parsing,
command execution, and output generation without requiring access to the
actual goldstandard dataset.
"""

import os
import sys
import json
import argparse
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from tempfile import TemporaryDirectory

# Import the main module to test
import expertise.benchmark.goldstandard_evaluator as evaluator_module
from expertise.benchmark.goldstandard_evaluator import (
    AffinityModelEvaluator,
    OpenReviewModelEvaluator,
    ExternalModelEvaluator,
    main
)

# ----- Mock Data for CLI Tests -----

# Sample evaluation results that would be returned by an evaluator
MOCK_EVALUATION_RESULTS = {
    "summary": {
        "model": "test_model",
        "loss": 0.15,
        "easy_triples": 0.88,
        "hard_triples": 0.65,
        "confidence_interval": {
            "loss": [0.12, 0.18],
            "easy_triples": [0.85, 0.91],
            "hard_triples": [0.61, 0.69]
        }
    },
    "iterations": {
        "d_20_1": {
            "loss": 0.16,
            "easy_triples": 0.87,
            "hard_triples": 0.64
        },
        "d_20_2": {
            "loss": 0.14,
            "easy_triples": 0.89,
            "hard_triples": 0.66
        }
    }
}

# ----- Tests for ArgumentParser -----

def test_argument_parser():
    """
    Test that the argument parser is correctly configured in main().
    
    This test verifies that:
    1. The parser defines the correct arguments
    2. Required arguments are marked as required
    3. Choices are properly defined for enumerated arguments
    """
    # Mock ArgumentParser and capture how it's called
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        # Setup mock parser and add_argument method
        mock_parser = MagicMock()
        MockArgumentParser.return_value = mock_parser
        mock_parser.add_argument = MagicMock()
        
        # Mock parse_args to avoid proceeding further
        mock_parser.parse_args.return_value = MagicMock()
        
        # Run the main function to setup the parser
        with patch('expertise.benchmark.goldstandard_evaluator.OpenReviewModelEvaluator'):
            try:
                main()
            except:
                pass  # We expect an error since we're not providing all required args
        
        # Verify parser was created with correct description
        MockArgumentParser.assert_called_once()
        
        # Verify required arguments were added
        add_argument_calls = mock_parser.add_argument.call_args_list
        required_args = ['--goldstandard_dir', '--output_dir', '--model_name', '--mode']
        
        for arg in required_args:
            matching_calls = [c for c in add_argument_calls if c[0][0] == arg]
            assert len(matching_calls) > 0, f"Required argument {arg} was not added to parser"
            assert matching_calls[0][1].get('required', False), f"Argument {arg} should be required"
        
        # Verify choices for mode argument
        mode_calls = [c for c in add_argument_calls if c[0][0] == '--mode']
        assert len(mode_calls) > 0, "Mode argument was not added to parser"
        assert 'choices' in mode_calls[0][1], "Mode argument should have choices defined"
        assert set(mode_calls[0][1]['choices']) == {'openreview', 'external'}, \
            "Mode choices should be 'openreview' and 'external'"

# ----- Tests for Main Function -----

def test_main_openreview_mode():
    """
    Test the main function with OpenReview mode.
    
    This test verifies that:
    1. The correct evaluator class is instantiated with the right parameters
    2. The evaluation is executed with the right config path
    3. Results are properly printed and displayed
    """
    # Create a mock parser and args
    mock_args = MagicMock()
    mock_args.goldstandard_dir = "/path/to/goldstandard"
    mock_args.output_dir = "/path/to/output"
    mock_args.model_name = "test_model"
    mock_args.mode = "openreview"
    mock_args.model_type = "bm25"
    mock_args.config_path = "/path/to/config.json"
    mock_args.predictions_dir = None
    
    # Mock the ArgumentParser
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser = MagicMock()
        MockArgumentParser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        
        # Mock the OpenReviewModelEvaluator class
        with patch('expertise.benchmark.goldstandard_evaluator.OpenReviewModelEvaluator') as MockEvaluator:
            # Configure the mock evaluator to return predefined results
            mock_evaluator_instance = MagicMock()
            mock_evaluator_instance.run_evaluation.return_value = MOCK_EVALUATION_RESULTS
            MockEvaluator.return_value = mock_evaluator_instance
            
            # Mock print to capture output
            with patch('builtins.print') as mock_print:
                # Run the main function
                main()
        
        # Verify the correct evaluator was instantiated with proper parameters
        MockEvaluator.assert_called_once_with(
            goldstandard_dir=mock_args.goldstandard_dir,
            output_dir=mock_args.output_dir,
            model_name=mock_args.model_name,
            model_type=mock_args.model_type
        )
        
        # Verify run_evaluation was called with the config path
        mock_evaluator_instance.run_evaluation.assert_called_once_with(mock_args.config_path)
        
        # Verify results were printed
        mock_print.assert_any_call("\nEvaluation Summary:")

def test_main_external_mode():
    """
    Test the main function with external mode.
    
    This test verifies that:
    1. The ExternalModelEvaluator is instantiated with the right parameters
    2. The evaluate_model method is called with the predictions directory
    3. Results are properly printed
    """
    # Create a mock parser and args
    mock_args = MagicMock()
    mock_args.goldstandard_dir = "/path/to/goldstandard"
    mock_args.output_dir = "/path/to/output"
    mock_args.model_name = "external_model"
    mock_args.mode = "external"
    mock_args.model_type = None
    mock_args.config_path = None
    mock_args.predictions_dir = "/path/to/predictions"
    
    # Mock the ArgumentParser
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser = MagicMock()
        MockArgumentParser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        
        # Mock the ExternalModelEvaluator class
        with patch('expertise.benchmark.goldstandard_evaluator.ExternalModelEvaluator') as MockEvaluator:
            # Configure the mock evaluator to return predefined results
            mock_evaluator_instance = MagicMock()
            mock_evaluator_instance.evaluate_model.return_value = MOCK_EVALUATION_RESULTS
            MockEvaluator.return_value = mock_evaluator_instance
            
            # Mock print to capture output
            with patch('builtins.print') as mock_print:
                # Run the main function
                main()
        
        # Verify the correct evaluator was instantiated with proper parameters
        MockEvaluator.assert_called_once_with(
            goldstandard_dir=mock_args.goldstandard_dir,
            output_dir=mock_args.output_dir,
            model_name=mock_args.model_name
        )
        
        # Verify evaluate_model was called with the predictions directory
        mock_evaluator_instance.evaluate_model.assert_called_once_with(mock_args.predictions_dir)
        
        # Verify results were printed
        mock_print.assert_any_call("\nEvaluation Summary:")

def test_main_argument_validation():
    """
    Test the argument validation in main function.
    
    This test verifies that:
    1. An error is raised if model_type is not provided in openreview mode
    2. An error is raised if predictions_dir is not provided in external mode
    """
    # Test missing model_type in openreview mode
    mock_args = MagicMock()
    mock_args.mode = "openreview"
    mock_args.model_type = None
    mock_args.goldstandard_dir = "/path/to/goldstandard"
    mock_args.output_dir = "/path/to/output"
    mock_args.model_name = "test_model"
    
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser = MagicMock()
        MockArgumentParser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        mock_parser.error = MagicMock()
        
        # Mock the OpenReviewModelEvaluator completely so it's never instantiated
        with patch('expertise.benchmark.goldstandard_evaluator.OpenReviewModelEvaluator', MagicMock()):
            # Run the main function
            main()
            
            # Verify parser.error was called
            mock_parser.error.assert_called_with("--model_type is required for openreview mode")
    
    # Test missing predictions_dir in external mode
    mock_args = MagicMock()
    mock_args.mode = "external"
    mock_args.predictions_dir = None
    mock_args.goldstandard_dir = "/path/to/goldstandard"
    mock_args.output_dir = "/path/to/output"
    mock_args.model_name = "test_model"
    
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser = MagicMock()
        MockArgumentParser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        mock_parser.error = MagicMock()
        
        # Mock the ExternalModelEvaluator completely so it's never instantiated
        with patch('expertise.benchmark.goldstandard_evaluator.ExternalModelEvaluator', MagicMock()):
            # Run the main function
            main()
            
            # Verify parser.error was called
            mock_parser.error.assert_called_with("--predictions_dir is required for external mode")

def test_results_output():
    """
    Test the printing of evaluation results.
    
    This test verifies that:
    1. The correct summary information is printed
    2. Different sections of the summary are properly displayed
    3. Error handling for missing data is appropriate
    """
    # Create a mock parser and args
    mock_args = MagicMock()
    mock_args.goldstandard_dir = "/path/to/goldstandard"
    mock_args.output_dir = "/path/to/output"
    mock_args.model_name = "test_model"
    mock_args.mode = "openreview"
    mock_args.model_type = "bm25"
    mock_args.config_path = "/path/to/config.json"
    
    # Create test results with some missing data to test robustness
    test_results = {
        "summary": {
            "model": "test_model",
            "loss": 0.15,
            "easy_triples": 0.88,
            "hard_triples": 0.65,
            "iterations_completed": 8,
            "completed_iterations": [1, 2, 3, 5, 6, 8, 9, 10],
            "confidence_interval": {
                "loss": [0.12, 0.18]
            }
        },
        "iterations": {
            "d_20_1": {"loss": 0.16},
            "d_20_2": {"loss": 0.14}
        }
    }
    
    # Mock the ArgumentParser
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser = MagicMock()
        MockArgumentParser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        
        # Mock the OpenReviewModelEvaluator
        with patch('expertise.benchmark.goldstandard_evaluator.OpenReviewModelEvaluator') as MockEvaluator:
            mock_evaluator = MagicMock()
            mock_evaluator.run_evaluation.return_value = test_results
            MockEvaluator.return_value = mock_evaluator
            
            # Mock print to capture output
            with patch('builtins.print') as mock_print:
                # Run the main function
                main()
                
            # Verify summary information was printed
            expected_calls = [
                call("\nEvaluation Summary:"),
                call("Model: test_model"),
                call("Iterations completed: 8/10"),
                call("Completed: [1, 2, 3, 5, 6, 8, 9, 10]"),
                call("Missing: [4, 7]"),
                call("Loss: 0.15"),
                call("95% CI: {'loss': [0.12, 0.18]}"),
                call("Easy Triples Accuracy: 0.88"),
                call("Hard Triples Accuracy: 0.65")
            ]
            
            # Check that all expected calls were made
            for expected in expected_calls:
                assert expected in mock_print.call_args_list, f"Expected print call {expected} not found" 