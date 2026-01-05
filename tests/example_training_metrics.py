"""
Example demonstrating training set evaluation for cross-validated models.

This example shows how to:
1. Evaluate training set performance to check for overfitting
2. Use separate feature vectors for test set evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.ml.training import train_ml_models_cv
from molml_mcp.tools.ml.evaluation import evaluate_cv_models
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
import numpy as np
import pandas as pd


def example_training_validation_comparison():
    """
    Example: Compare training vs validation performance to detect overfitting.
    
    This is useful for understanding if your model is overfitting by comparing
    training set metrics (usually higher) with validation set metrics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Training vs Validation Performance")
    print("="*80)
    
    manifest_path = "tests/data/test_manifest.json"
    
    # Create synthetic classification dataset
    np.random.seed(42)
    n_samples = 100
    smiles = [f"CC(C)C{i}" for i in range(n_samples)]
    labels = np.random.randint(0, 2, n_samples)
    
    # Create dataset
    df = pd.DataFrame({
        "smiles": smiles,
        "label": labels
    })
    
    dataset_filename = _store_resource(
        df, manifest_path, "training_example_dataset",
        "Example dataset for training metrics", 'csv'
    )
    
    # Create random features
    features = {smi: np.random.randn(10).tolist() for smi in smiles}
    features_filename = _store_resource(
        features, manifest_path, "training_example_features",
        "Example features", 'json'
    )
    
    # Train CV models
    print("\n1. Training cross-validated models...")
    cv_result = train_ml_models_cv(
        input_filename=dataset_filename,
        feature_vectors_filename=features_filename,
        smiles_column="smiles",
        label_column="label",
        model_algorithm="random_forest_classifier",
        cv_strategy="kfold",
        n_folds=5,
        project_manifest_path=manifest_path,
        output_filename="cv_models_training_example",
        explanation="CV models for training metrics example"
    )
    
    # Evaluate with both training and validation metrics
    print("\n2. Evaluating training and validation sets...")
    eval_result = evaluate_cv_models(
        cv_model_filename=cv_result["output_filename"],
        feature_vectors_filename=features_filename,
        project_manifest_path=manifest_path,
        metrics=["accuracy", "f1_score"],
        output_filename="training_validation_comparison",
        explanation="Training vs validation comparison",
        use_validation_sets=True,
        evaluate_training_sets=True  # Enable training set evaluation
    )
    
    # Print comparison
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)
    
    for metric in ["accuracy", "f1_score"]:
        train_mean = eval_result['training_metrics_summary'][metric]['mean']
        train_std = eval_result['training_metrics_summary'][metric]['std']
        val_mean = eval_result['metrics_summary'][metric]['mean']
        val_std = eval_result['metrics_summary'][metric]['std']
        gap = train_mean - val_mean
        
        print(f"\n{metric.upper()}:")
        print(f"  Training:   {train_mean:.4f} ± {train_std:.4f}")
        print(f"  Validation: {val_mean:.4f} ± {val_std:.4f}")
        print(f"  Gap:        {gap:.4f} {'(possible overfitting)' if gap > 0.1 else '(good fit)'}")
    
    print("="*80)


def example_separate_test_features():
    """
    Example: Use separate feature vectors for test set evaluation.
    
    This is useful when test set features are computed separately or using
    a different featurization scheme.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Separate Test Set Feature Vectors")
    print("="*80)
    
    manifest_path = "tests/data/test_manifest.json"
    
    # Create training dataset
    np.random.seed(42)
    n_train = 80
    train_smiles = [f"CC(C)C{i}" for i in range(n_train)]
    train_labels = np.random.randint(0, 2, n_train)
    
    train_df = pd.DataFrame({
        "smiles": train_smiles,
        "label": train_labels
    })
    
    train_dataset_filename = _store_resource(
        train_df, manifest_path, "train_dataset_features_ex",
        "Training dataset", 'csv'
    )
    
    # Create test dataset
    n_test = 20
    test_smiles = [f"CC(C)C{i}" for i in range(n_train, n_train + n_test)]
    test_labels = np.random.randint(0, 2, n_test)
    
    test_df = pd.DataFrame({
        "smiles": test_smiles,
        "label": test_labels
    })
    
    test_dataset_filename = _store_resource(
        test_df, manifest_path, "test_dataset_features_ex",
        "Test dataset", 'csv'
    )
    
    # Create training features
    train_features = {smi: np.random.randn(10).tolist() for smi in train_smiles}
    train_features_filename = _store_resource(
        train_features, manifest_path, "train_features_ex",
        "Training features", 'json'
    )
    
    # Create SEPARATE test features (simulating different feature computation)
    test_features = {smi: np.random.randn(10).tolist() for smi in test_smiles}
    test_features_filename = _store_resource(
        test_features, manifest_path, "test_features_ex",
        "Test features (separate)", 'json'
    )
    
    # Train CV models
    print("\n1. Training cross-validated models...")
    cv_result = train_ml_models_cv(
        input_filename=train_dataset_filename,
        feature_vectors_filename=train_features_filename,
        smiles_column="smiles",
        label_column="label",
        model_algorithm="random_forest_classifier",
        cv_strategy="kfold",
        n_folds=5,
        project_manifest_path=manifest_path,
        output_filename="cv_models_separate_features",
        explanation="CV models for separate features example"
    )
    
    # Evaluate on test set with separate features
    print("\n2. Evaluating on test set with separate feature vectors...")
    eval_result = evaluate_cv_models(
        cv_model_filename=cv_result["output_filename"],
        feature_vectors_filename=train_features_filename,  # Training features (not used for test)
        project_manifest_path=manifest_path,
        metrics=["accuracy", "f1_score"],
        output_filename="test_eval_separate_features",
        explanation="Test set evaluation with separate features",
        use_validation_sets=False,  # Use test set
        test_input_filename=test_dataset_filename,
        test_feature_vectors_filename=test_features_filename,  # Use SEPARATE test features
        test_smiles_column="smiles",
        test_label_column="label"
    )
    
    print("\n✓ Successfully evaluated using separate test feature vectors!")
    print(f"  Test set accuracy: {eval_result['metrics_summary']['accuracy']['mean']:.4f}")


if __name__ == "__main__":
    example_training_validation_comparison()
    example_separate_test_features()
    print("\n✓ All examples completed successfully!\n")
