"""
Example usage of evaluate_cv_models function.

This script demonstrates how to evaluate cross-validated models trained
with train_ml_models_cv() to assess model performance and consistency.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from molml_mcp.tools.ml.training import train_ml_models_cv
from molml_mcp.tools.ml.evaluation import evaluate_cv_models
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

# Use existing test manifest
TEST_MANIFEST = Path(__file__).parent / "data" / "test_manifest.json"


def example_validation_set_evaluation():
    """Example 1: Evaluate models on their validation sets (CV performance)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Cross-Validation Performance Evaluation")
    print("="*80)
    
    # Create dataset for binary classification
    smiles_list = [
        'CCO', 'CC(C)O', 'CCCO', 'CCCCO', 'CC(C)CO',
        'CCCCCO', 'CC(C)CCO', 'CCCCCCO', 'CC(C)CCCO', 'CCCCCCCO',
        'CCC', 'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC',
        'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1'
    ]
    
    # Binary classification: contains oxygen (1) vs no oxygen (0)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # alcohols
              0, 0, 0, 0, 0,                  # alkanes
              0, 0, 0, 1, 1]                  # aromatics
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'has_oxygen': labels
    })
    
    # Simple features
    feature_vectors = {}
    for smi in smiles_list:
        n_carbons = smi.count('C')
        n_oxygens = smi.count('O')
        has_aromatic = int('c' in smi.lower())
        feature_vectors[smi] = [float(n_carbons), float(n_oxygens), float(has_aromatic)]
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'example_eval_data', 'Example data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'example_eval_features', 'Features', 'json')
    
    print(f"\nDataset: {len(df)} molecules")
    print(f"  - With oxygen: {sum(labels)}")
    print(f"  - Without oxygen: {len(labels) - sum(labels)}")
    
    # Train CV models using Monte Carlo strategy
    print("\nTraining 5 models with Monte Carlo CV...")
    train_result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='has_oxygen',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='example_cv_models',
        explanation='Example CV models for oxygen detection',
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 100, 'max_depth': 5},
        cv_strategy='montecarlo',
        n_folds=5,
        val_size=0.25,
        random_state=42
    )
    
    print(f"✓ Training complete: {train_result['n_models']} models trained")
    
    # Evaluate models on their validation sets
    print("\nEvaluating models on validation sets (CV performance)...")
    eval_result = evaluate_cv_models(
        cv_model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy', 'precision', 'recall', 'f1_score'],
        output_filename='example_cv_evaluation',
        use_validation_sets=True  # Evaluate on validation sets
    )
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Report: {eval_result['output_filename']}")
    
    # Display summary
    print("\nCross-Validation Performance Summary:")
    print("-" * 60)
    for metric_name, stats in eval_result['metrics_summary'].items():
        if stats['n_folds'] > 0:
            print(f"{metric_name:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"{'':15s}  (min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    # Load full report for more details
    report = _load_resource(str(TEST_MANIFEST), eval_result['output_filename'])
    
    print(f"\nPer-Fold Performance:")
    print("-" * 60)
    for fold_metrics in report['per_fold_metrics']:
        fold_num = fold_metrics['fold']
        accuracy = fold_metrics['metrics']['accuracy']
        n_samples = fold_metrics['n_samples']
        print(f"Fold {fold_num}: Accuracy = {accuracy:.4f} ({n_samples} validation samples)")


def example_test_set_evaluation():
    """Example 2: Evaluate all models on independent test set."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Test Set Evaluation (Model Consistency)")
    print("="*80)
    
    # Create train and test datasets
    train_smiles = ['C'*i for i in range(1, 16)]
    test_smiles = ['C'*i for i in range(16, 21)]
    
    # Regression: predict carbon count
    train_labels = [float(len(smi)) for smi in train_smiles]
    test_labels = [float(len(smi)) for smi in test_smiles]
    
    train_df = pd.DataFrame({
        'smiles': train_smiles,
        'carbon_count': train_labels
    })
    
    test_df = pd.DataFrame({
        'smiles': test_smiles,
        'carbon_count': test_labels
    })
    
    # Features
    all_smiles = train_smiles + test_smiles
    feature_vectors = {smi: [float(len(smi)), float(len(smi))**2] for smi in all_smiles}
    
    # Store data
    train_file = _store_resource(train_df, str(TEST_MANIFEST), 'example_train_data', 'Train data', 'csv')
    test_file = _store_resource(test_df, str(TEST_MANIFEST), 'example_test_data', 'Test data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'example_features', 'Features', 'json')
    
    print(f"\nTrain set: {len(train_df)} molecules")
    print(f"Test set:  {len(test_df)} molecules")
    
    # Train CV models
    print("\nTraining 10 models with Monte Carlo CV...")
    train_result = train_ml_models_cv(
        input_filename=train_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='carbon_count',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='example_regression_cv_models',
        explanation='Regression CV models',
        model_algorithm='random_forest_regressor',
        cv_strategy='montecarlo',
        n_folds=10,
        val_size=0.2,
        random_state=42
    )
    
    print(f"✓ Training complete: {train_result['n_models']} models trained")
    
    # Evaluate all models on the same test set
    print("\nEvaluating all models on independent test set...")
    eval_result = evaluate_cv_models(
        cv_model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['mse', 'mae', 'r2'],
        output_filename='example_test_evaluation',
        use_validation_sets=False,  # Use independent test set
        test_input_filename=test_file,
        test_smiles_column='smiles',
        test_label_column='carbon_count'
    )
    
    print(f"\n✓ Evaluation complete!")
    
    # Display summary
    print("\nTest Set Performance Summary:")
    print("-" * 60)
    for metric_name, stats in eval_result['metrics_summary'].items():
        if stats['n_folds'] > 0:
            print(f"{metric_name:10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"{'':10s}  (min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    print("\nInterpretation:")
    print("  - Mean: Average performance across all models")
    print("  - Std:  Model consistency (lower = more consistent)")
    print("  - For ensemble predictions, you can average all model outputs")


def example_comparing_cv_strategies():
    """Example 3: Compare different CV strategies."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing CV Strategies")
    print("="*80)
    
    # Create dataset
    smiles_list = ['C'*i + 'O' for i in range(1, 21)]
    labels = [0]*10 + [1]*10  # Binary: short vs long chains
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'label': labels
    })
    
    feature_vectors = {smi: [float(len(smi)), float(smi.count('C'))] for smi in smiles_list}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'example_compare_data', 'Data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'example_compare_features', 'Features', 'json')
    
    print(f"\nDataset: {len(df)} molecules")
    
    strategies = [
        ('kfold', 'K-Fold (random)'),
        ('stratified', 'Stratified K-Fold'),
        ('montecarlo', 'Monte Carlo')
    ]
    
    results = {}
    
    for strategy_name, strategy_label in strategies:
        print(f"\n{'-'*80}")
        print(f"Training and evaluating: {strategy_label}")
        print(f"{'-'*80}")
        
        # Train models
        train_result = train_ml_models_cv(
            input_filename=data_file,
            feature_vectors_filename=features_file,
            smiles_column='smiles',
            label_column='label',
            project_manifest_path=str(TEST_MANIFEST),
            output_filename=f'example_cv_{strategy_name}',
            explanation=f'{strategy_label} models',
            model_algorithm='random_forest_classifier',
            cv_strategy=strategy_name,
            n_folds=5,
            val_size=0.2 if strategy_name == 'montecarlo' else None,
            random_state=42
        )
        
        # Evaluate
        eval_result = evaluate_cv_models(
            cv_model_filename=train_result['output_filename'],
            feature_vectors_filename=features_file,
            project_manifest_path=str(TEST_MANIFEST),
            metrics=['accuracy', 'f1_score'],
            output_filename=f'example_eval_{strategy_name}',
            use_validation_sets=True
        )
        
        results[strategy_label] = eval_result['metrics_summary']
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON OF CV STRATEGIES")
    print("="*80)
    print(f"\n{'Strategy':<25} {'Accuracy':<20} {'F1 Score':<20}")
    print("-" * 65)
    
    for strategy_label in [label for _, label in strategies]:
        acc_stats = results[strategy_label]['accuracy']
        f1_stats = results[strategy_label]['f1_score']
        
        acc_str = f"{acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}"
        f1_str = f"{f1_stats['mean']:.4f} ± {f1_stats['std']:.4f}"
        
        print(f"{strategy_label:<25} {acc_str:<20} {f1_str:<20}")
    
    print("\nKey Insights:")
    print("  - K-Fold: Standard CV, each sample used once for validation")
    print("  - Stratified: Maintains class balance in each fold")
    print("  - Monte Carlo: Flexible validation size, repeated sampling")
    print("  - Lower std = more consistent performance across folds")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("EVALUATE_CV_MODELS EXAMPLES")
    print("="*80)
    print("\nDemonstrating cross-validated model evaluation strategies.")
    
    example_validation_set_evaluation()
    example_test_set_evaluation()
    example_comparing_cv_strategies()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Validation Set Eval: Assess CV performance (model generalization)")
    print("  2. Test Set Eval: Assess model consistency on unseen data")
    print("  3. Compare Strategies: Different CV methods may suit different problems")
    print("\nUse Cases:")
    print("  - Model Selection: Choose best performing model from CV")
    print("  - Ensemble Building: Combine predictions from all models")
    print("  - Performance Reporting: Report mean ± std for robustness")
    print("  - Debugging: Identify problematic folds or data issues")
    print()
