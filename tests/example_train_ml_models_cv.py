"""
Example usage of train_ml_models_cv function.

This script demonstrates how to use cross-validation training to create
multiple models for ensemble predictions or robust evaluation.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from molml_mcp.tools.ml.training import train_ml_models_cv
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

# Use existing test manifest
TEST_MANIFEST = Path(__file__).parent / "data" / "test_manifest.json"


def example_monte_carlo_cv():
    """Example: Monte Carlo Cross-Validation."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Monte Carlo Cross-Validation")
    print("="*80)
    
    # Create example dataset (simple molecular property prediction)
    smiles_list = [
        'CCO', 'CC(C)O', 'CCCO', 'CCCCO', 'CC(C)CO',
        'CCCCCO', 'CC(C)CCO', 'CCCCCCO', 'CC(C)CCCO', 'CCCCCCCO',
        'CCC', 'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC',
        'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1'
    ]
    
    # Binary classification: alcohols (1) vs non-alcohols (0)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # alcohols
              0, 0, 0, 0, 0,                  # alkanes
              0, 0, 0, 1, 1]                  # aromatics (some with OH/NH2)
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'label': labels
    })
    
    # Create simple feature vectors (carbon count, oxygen count, has aromatic ring)
    feature_vectors = {}
    for smi in smiles_list:
        n_carbons = smi.count('C')
        n_oxygens = smi.count('O')
        has_aromatic = int('c' in smi.lower() and 'cc' in smi.lower())
        feature_vectors[smi] = [float(n_carbons), float(n_oxygens), float(has_aromatic)]
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'example_cv_data', 'Example CV data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'example_cv_features', 'Example CV features', 'json')
    
    print(f"\nDataset: {len(df)} molecules")
    print(f"  - Alcohols: {sum(labels)}")
    print(f"  - Non-alcohols: {len(labels) - sum(labels)}")
    
    # Train models with Monte Carlo CV
    print("\nTraining 5 models with Monte Carlo CV (20% validation size)...")
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='example_montecarlo_models',
        explanation='Example Monte Carlo CV models for alcohol classification',
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 100, 'max_depth': 5},
        cv_strategy='montecarlo',
        n_folds=5,
        val_size=0.2,
        random_state=42
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Model file: {result['output_filename']}")
    print(f"  Algorithm: {result['model_algorithm']}")
    print(f"  Number of models: {result['n_models']}")
    print(f"  Features per model: {result['n_features']}")
    
    # Load and inspect the models
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    print(f"\nModel structure:")
    print(f"  - models: {len(model_data['models'])} trained classifiers")
    print(f"  - data_splits: {len(model_data['data_splits'])} train/val splits")
    
    for i, split in enumerate(model_data['data_splits']):
        n_train = len(split['training'])
        n_val = len(split['validation'])
        print(f"  - Split {i+1}: {n_train} train, {n_val} val")


def example_stratified_kfold():
    """Example: Stratified K-Fold Cross-Validation."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Stratified K-Fold Cross-Validation")
    print("="*80)
    
    # Create imbalanced dataset
    smiles_list = ['C'*i + 'O' for i in range(1, 16)]  # C1O, C2O, ..., C15O
    
    # Imbalanced classification: short chains (0) vs long chains (1)
    # 10 samples of class 0, 5 samples of class 1
    labels = [0]*10 + [1]*5
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'label': labels
    })
    
    # Simple features
    feature_vectors = {smi: [float(smi.count('C')), 1.0, float(smi.count('C'))**2] 
                      for smi in smiles_list}
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'example_stratified_data', 'Example stratified data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'example_stratified_features', 'Example stratified features', 'json')
    
    print(f"\nImbalanced dataset: {len(df)} molecules")
    print(f"  - Class 0: {sum(1 for x in labels if x == 0)} samples")
    print(f"  - Class 1: {sum(1 for x in labels if x == 1)} samples")
    print(f"  - Imbalance ratio: {sum(labels) / len(labels):.2f}")
    
    # Train with stratified K-fold to maintain class distribution
    print("\nTraining 5 models with Stratified K-Fold CV...")
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='example_stratified_models',
        explanation='Example Stratified K-Fold models',
        model_algorithm='random_forest_classifier',
        cv_strategy='stratified',
        n_folds=5,
        random_state=42
    )
    
    print(f"\n✓ Stratified training complete!")
    print(f"  Maintains class distribution in each fold")
    
    # Verify stratification
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    print(f"\nClass distribution per fold:")
    for i, split in enumerate(model_data['data_splits']):
        val_labels = list(split['validation'].values())
        n_class_0 = sum(1 for x in val_labels if x == 0)
        n_class_1 = sum(1 for x in val_labels if x == 1)
        ratio = n_class_1 / len(val_labels) if len(val_labels) > 0 else 0
        print(f"  - Fold {i+1}: Class 0={n_class_0}, Class 1={n_class_1}, Ratio={ratio:.2f}")


def example_scaffold_based_cv():
    """Example: Scaffold-Based Cross-Validation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Scaffold-Based Cross-Validation")
    print("="*80)
    
    # Create dataset with molecules grouped by scaffold
    # Scaffold 1: simple alkanes
    scaffold1_smiles = ['CCC', 'CCCC', 'CCCCC']
    # Scaffold 2: alcohols
    scaffold2_smiles = ['CCO', 'CCCO', 'CCCCO']
    # Scaffold 3: amines
    scaffold3_smiles = ['CCN', 'CCCN', 'CCCCN']
    # Scaffold 4: aromatics
    scaffold4_smiles = ['c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1']
    
    all_smiles = scaffold1_smiles + scaffold2_smiles + scaffold3_smiles + scaffold4_smiles
    
    # Scaffold identifiers
    scaffolds = (['alkane']*3 + ['alcohol']*3 + ['amine']*3 + ['aromatic']*3)
    
    # Binary classification: polar (1) vs non-polar (0)
    labels = [0]*3 + [1]*3 + [1]*3 + [0]*3
    
    df = pd.DataFrame({
        'smiles': all_smiles,
        'scaffold': scaffolds,
        'label': labels
    })
    
    # Simple features
    feature_vectors = {smi: [float(len(smi)), float('O' in smi or 'N' in smi), float('c' in smi)] 
                      for smi in all_smiles}
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'example_scaffold_data', 'Example scaffold data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'example_scaffold_features', 'Example scaffold features', 'json')
    
    print(f"\nScaffold-based dataset: {len(df)} molecules across 4 scaffolds")
    print(f"  - Alkanes: {sum(1 for s in scaffolds if s == 'alkane')} molecules")
    print(f"  - Alcohols: {sum(1 for s in scaffolds if s == 'alcohol')} molecules")
    print(f"  - Amines: {sum(1 for s in scaffolds if s == 'amine')} molecules")
    print(f"  - Aromatics: {sum(1 for s in scaffolds if s == 'aromatic')} molecules")
    
    # Train with scaffold-based CV (molecules with same scaffold stay together)
    print("\nTraining 4 models with Scaffold-Based CV...")
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='example_scaffold_models',
        explanation='Example Scaffold-Based CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='scaffold',
        scaffold_column='scaffold',
        n_folds=4,  # One fold per scaffold
        random_state=42
    )
    
    print(f"\n✓ Scaffold-based training complete!")
    print(f"  Ensures no scaffold appears in both train and validation")
    
    # Verify scaffold separation
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    smiles_to_scaffold = dict(zip(df['smiles'], df['scaffold']))
    
    print(f"\nScaffold separation per fold:")
    for i, split in enumerate(model_data['data_splits']):
        train_scaffolds = set(smiles_to_scaffold[smi] for smi in split['training'].keys())
        val_scaffolds = set(smiles_to_scaffold[smi] for smi in split['validation'].keys())
        print(f"  - Fold {i+1}:")
        print(f"      Train scaffolds: {sorted(train_scaffolds)}")
        print(f"      Val scaffolds: {sorted(val_scaffolds)}")
        print(f"      Overlap: {train_scaffolds & val_scaffolds} (should be empty)")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TRAIN_ML_MODELS_CV EXAMPLES")
    print("="*80)
    print("\nDemonstrating different cross-validation strategies for model training.")
    
    example_monte_carlo_cv()
    example_stratified_kfold()
    example_scaffold_based_cv()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey takeaways:")
    print("  1. Monte Carlo CV: Flexible validation size, repeated random sampling")
    print("  2. Stratified K-Fold: Maintains class distribution in each fold")
    print("  3. Scaffold-Based CV: Ensures models generalize to new scaffolds")
    print("\nAll trained models are stored and can be used for:")
    print("  - Ensemble predictions (average predictions from all models)")
    print("  - Cross-validation evaluation (assess generalization)")
    print("  - Model selection (compare different hyperparameters)")
    print()
