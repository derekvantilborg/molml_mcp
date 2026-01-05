"""Test train_ml_models_cv function."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from molml_mcp.tools.ml.training import train_ml_models_cv
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

# Use existing test manifest
TEST_MANIFEST = Path(__file__).parent / "data" / "test_manifest.json"


@pytest.fixture
def test_data():
    """Create test dataset and feature vectors."""
    # Create simple test data
    smiles_list = [
        'CCO',      # ethanol
        'CC(C)O',   # isopropanol
        'CCCO',     # propanol
        'CCCCO',    # butanol
        'CC(C)CO',  # isobutanol
        'CCCCCO',   # pentanol
        'CC(C)CCO', # isopentanol
        'CCCCCCO',  # hexanol
        'CC(C)CCCO',# 4-methyl-1-pentanol
        'CCCCCCCO', # heptanol
    ]
    
    # Binary classification: alcohols with <4 carbons (0) vs >=4 carbons (1)
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
    # Create DataFrame
    df = pd.DataFrame({
        'smiles': smiles_list,
        'label': labels
    })
    
    # Create simple feature vectors (molecular weight, carbon count, oxygen count)
    feature_vectors = {}
    for smi in smiles_list:
        n_carbons = smi.count('C')
        n_oxygens = smi.count('O')
        # Simple features: [n_carbons, n_oxygens, n_carbons^2]
        feature_vectors[smi] = [float(n_carbons), float(n_oxygens), float(n_carbons**2)]
    
    return df, feature_vectors


def test_train_ml_models_cv_montecarlo(test_data):
    """Test Monte Carlo CV strategy."""
    df, feature_vectors = test_data
    
    # Store test data
    data_file = _store_resource(
        df,
        str(TEST_MANIFEST),
        'cv_test_data',
        'Test data for CV',
        'csv'
    )
    
    features_file = _store_resource(
        feature_vectors,
        str(TEST_MANIFEST),
        'cv_test_features',
        'Test features for CV',
        'json'
    )
    
    # Train models with Monte Carlo CV
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_montecarlo',
        explanation='Test Monte Carlo CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='montecarlo',
        n_folds=3,
        val_size=0.3,
        random_state=42
    )
    
    # Verify result structure
    assert 'output_filename' in result
    assert result['model_algorithm'] == 'random_forest_classifier'
    assert result['n_models'] == 3
    assert result['n_features'] == 3
    assert result['cv_strategy'] == 'montecarlo'
    
    # Load and verify stored models
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    assert 'models' in model_data
    assert 'data_splits' in model_data
    assert len(model_data['models']) == 3
    assert len(model_data['data_splits']) == 3
    
    # Check each model and data split
    for i, (model, split) in enumerate(zip(model_data['models'], model_data['data_splits'])):
        assert model is not None
        assert 'training' in split
        assert 'validation' in split
        assert len(split['training']) > 0
        assert len(split['validation']) > 0
        
        # Verify train/val split doesn't overlap
        train_smiles = set(split['training'].keys())
        val_smiles = set(split['validation'].keys())
        assert len(train_smiles & val_smiles) == 0, f"Split {i}: train and val overlap"


def test_train_ml_models_cv_kfold(test_data):
    """Test K-Fold CV strategy."""
    df, feature_vectors = test_data
    
    # Store test data
    data_file = _store_resource(
        df,
        str(TEST_MANIFEST),
        'cv_test_data_kfold',
        'Test data for K-fold CV',
        'csv'
    )
    
    features_file = _store_resource(
        feature_vectors,
        str(TEST_MANIFEST),
        'cv_test_features_kfold',
        'Test features for K-fold CV',
        'json'
    )
    
    # Train models with K-Fold CV
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_kfold',
        explanation='Test K-Fold CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='kfold',
        n_folds=5,
        random_state=42
    )
    
    # Verify result
    assert result['n_models'] == 5
    assert result['cv_strategy'] == 'kfold'
    
    # Load and verify
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    assert len(model_data['models']) == 5
    
    # In K-fold, all samples should appear in validation exactly once
    all_val_smiles = set()
    for split in model_data['data_splits']:
        val_smiles = set(split['validation'].keys())
        # No overlap across folds
        assert len(all_val_smiles & val_smiles) == 0
        all_val_smiles.update(val_smiles)
    
    # All samples should be covered
    assert len(all_val_smiles) == len(df)


def test_train_ml_models_cv_stratified(test_data):
    """Test Stratified K-Fold CV strategy."""
    df, feature_vectors = test_data
    
    # Store test data
    data_file = _store_resource(
        df,
        str(TEST_MANIFEST),
        'cv_test_data_stratified',
        'Test data for stratified CV',
        'csv'
    )
    
    features_file = _store_resource(
        feature_vectors,
        str(TEST_MANIFEST),
        'cv_test_features_stratified',
        'Test features for stratified CV',
        'json'
    )
    
    # Train models with Stratified K-Fold CV
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_stratified',
        explanation='Test Stratified K-Fold CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='stratified',
        n_folds=2,  # Use 2 folds for small dataset
        random_state=42
    )
    
    # Verify result
    assert result['n_models'] == 2
    assert result['cv_strategy'] == 'stratified'
    
    # Load and verify class distribution is maintained
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    # Check that class distribution is roughly maintained in each validation set
    original_pos_ratio = df['label'].mean()
    
    for split in model_data['data_splits']:
        val_labels = list(split['validation'].values())
        val_pos_ratio = sum(val_labels) / len(val_labels)
        
        # Stratified split should maintain similar class distribution
        # Allow 20% deviation due to small sample size
        assert abs(val_pos_ratio - original_pos_ratio) < 0.3


def test_train_ml_models_cv_scaffold(test_data):
    """Test Scaffold-based CV strategy."""
    df, feature_vectors = test_data
    
    # Add scaffold column (simple scaffold based on carbon chain length)
    # Molecules with same carbon count share a scaffold
    df['scaffold'] = df['smiles'].apply(lambda s: f"C{s.count('C')}")
    
    # Store test data
    data_file = _store_resource(
        df,
        str(TEST_MANIFEST),
        'cv_test_data_scaffold',
        'Test data for scaffold CV',
        'csv'
    )
    
    features_file = _store_resource(
        feature_vectors,
        str(TEST_MANIFEST),
        'cv_test_features_scaffold',
        'Test features for scaffold CV',
        'json'
    )
    
    # Train models with Scaffold CV
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_scaffold',
        explanation='Test Scaffold CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='scaffold',
        scaffold_column='scaffold',
        n_folds=2,  # Limited by number of unique scaffolds
        random_state=42
    )
    
    # Verify result
    assert result['cv_strategy'] == 'scaffold'
    assert result['n_models'] >= 2
    
    # Load and verify scaffold separation
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    # Create scaffold mapping
    smiles_to_scaffold = dict(zip(df['smiles'], df['scaffold']))
    
    # Verify no scaffold appears in both train and val for same split
    for split in model_data['data_splits']:
        train_scaffolds = set(smiles_to_scaffold[smi] for smi in split['training'].keys())
        val_scaffolds = set(smiles_to_scaffold[smi] for smi in split['validation'].keys())
        
        # No scaffold overlap between train and val
        assert len(train_scaffolds & val_scaffolds) == 0


def test_train_ml_models_cv_hyperparameters(test_data):
    """Test that hyperparameters are properly passed."""
    df, feature_vectors = test_data
    
    # Store test data
    data_file = _store_resource(
        df,
        str(TEST_MANIFEST),
        'cv_test_data_hyperparams',
        'Test data for hyperparameter CV',
        'csv'
    )
    
    features_file = _store_resource(
        feature_vectors,
        str(TEST_MANIFEST),
        'cv_test_features_hyperparams',
        'Test features for hyperparameter CV',
        'json'
    )
    
    # Train models with custom hyperparameters
    custom_hyperparams = {
        'n_estimators': 50,
        'max_depth': 3
    }
    
    result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_hyperparams',
        explanation='Test CV with hyperparameters',
        model_algorithm='random_forest_classifier',
        hyperparameters=custom_hyperparams,
        cv_strategy='kfold',
        n_folds=2,
        random_state=42
    )
    
    # Verify hyperparameters are stored
    assert result['hyperparameters'] == custom_hyperparams
    
    # Load and verify models have the hyperparameters
    model_data = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    assert model_data['hyperparameters'] == custom_hyperparams
    
    # Check that models have the expected parameters
    for model in model_data['models']:
        assert hasattr(model, 'n_estimators')
        assert model.n_estimators == 50
        assert hasattr(model, 'max_depth')
        assert model.max_depth == 3


def test_train_ml_models_cv_missing_columns():
    """Test error handling for missing columns."""
    df = pd.DataFrame({'wrong_col': [1, 2, 3]})
    feature_vectors = {'A': [1.0], 'B': [2.0], 'C': [3.0]}
    
    data_file = _store_resource(
        df,
        str(TEST_MANIFEST),
        'cv_test_error_data',
        'Test error handling',
        'csv'
    )
    
    features_file = _store_resource(
        feature_vectors,
        str(TEST_MANIFEST),
        'cv_test_error_features',
        'Test error handling',
        'json'
    )
    
    with pytest.raises(ValueError, match="SMILES column.*not found"):
        train_ml_models_cv(
            input_filename=data_file,
            feature_vectors_filename=features_file,
            smiles_column='smiles',
            label_column='label',
            project_manifest_path=str(TEST_MANIFEST),
            output_filename='cv_models_error',
            explanation='Should fail',
            cv_strategy='kfold',
            n_folds=2
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
