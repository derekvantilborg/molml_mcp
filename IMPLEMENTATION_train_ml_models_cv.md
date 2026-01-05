# train_ml_models_cv Implementation Summary

## Overview
Implemented `train_ml_models_cv()` function in `src/molml_mcp/tools/ml/training.py` to support cross-validation-based model training with all CV strategies from `cross_validation.py`.

## Key Features

### Supported CV Strategies
1. **K-Fold** (`kfold`): Standard K-fold cross-validation
2. **Stratified K-Fold** (`stratified`): Maintains class distribution in each fold
3. **Monte Carlo** (`montecarlo`): Repeated random sub-sampling validation
4. **Scaffold-Based** (`scaffold`): Groups molecules by chemical scaffold
5. **Cluster-Based** (`cluster`): Groups molecules by cluster assignment
6. **Leave-P-Out** (`leavepout`): Leaves out P samples for validation

### Function Signature
```python
def train_ml_models_cv(
    input_filename: str,                    # CSV with SMILES and labels
    feature_vectors_filename: str,          # JSON with SMILES -> features
    smiles_column: str,                     # Column name for SMILES
    label_column: str,                      # Column name for labels
    project_manifest_path: str,             # Path to manifest.json
    output_filename: str,                   # Output model name
    explanation: str,                       # Description
    model_algorithm: str = "random_forest_classifier",
    hyperparameters: dict = None,
    cv_strategy: str = "montecarlo",
    n_folds: int = 5,
    val_size: Optional[float] = None,
    cluster_column: Optional[str] = None,
    scaffold_column: Optional[str] = None,
    shuffle: bool = True,
    p: int = 1,
    max_splits: Optional[int] = None,
    random_state: int = 42
) -> dict
```

### Output Structure
The function stores a model resource with the following structure:
```python
{
    "models": [model1, model2, ...],        # List of trained models
    "data_splits": [                        # List of train/val splits
        {
            "training": {smi: label, ...},  # Training SMILES -> label mapping
            "validation": {smi: label, ...} # Validation SMILES -> label mapping
        },
        ...
    ],
    "model_algorithm": str,
    "hyperparameters": dict,
    "random_state": int,
    "n_features": int,
    "cv_strategy": str,
    "cv_parameters": {
        "n_folds": int,
        "val_size": float,
        ...
    }
}
```

### Return Value
```python
{
    "output_filename": str,      # Saved model filename
    "model_algorithm": str,      # Algorithm used
    "n_models": int,             # Number of models trained
    "n_features": int,           # Number of features per model
    "cv_strategy": str,          # CV strategy used
    "n_folds": int,              # Number of folds/splits
    "hyperparameters": dict      # Hyperparameters used
}
```

## Integration

### Module Exports
Updated `src/molml_mcp/tools/ml/__init__.py`:
- Added `train_ml_models_cv` to imports
- Added to `get_all_ml_tools()` return list
- Function is now automatically registered with MCP server

### MCP Server Registration
The function is automatically registered through the existing server setup:
```python
# In server.py
from molml_mcp.tools.ml import get_all_ml_tools
for tool_func in get_all_ml_tools():
    mcp.add_tool(tool_func)
```

## Testing

### Test Coverage
Created comprehensive test suite in `tests/test_train_ml_models_cv.py`:

1. ✅ `test_train_ml_models_cv_montecarlo` - Monte Carlo CV
2. ✅ `test_train_ml_models_cv_kfold` - K-Fold CV
3. ✅ `test_train_ml_models_cv_stratified` - Stratified K-Fold CV
4. ✅ `test_train_ml_models_cv_scaffold` - Scaffold-based CV
5. ✅ `test_train_ml_models_cv_hyperparameters` - Hyperparameter passing
6. ✅ `test_train_ml_models_cv_missing_columns` - Error handling

**All 6 tests pass successfully**

### Example Usage
Created `tests/example_train_ml_models_cv.py` with three complete examples:
1. Monte Carlo CV for alcohol classification
2. Stratified K-Fold CV for imbalanced datasets
3. Scaffold-based CV for chemical diversity

## Use Cases

### Ensemble Predictions
Train multiple models on different CV splits and average their predictions for more robust results.

### Cross-Validation Evaluation
Evaluate model performance across multiple train/validation splits to assess generalization.

### Scaffold-Based Evaluation
Test if models can generalize to new chemical scaffolds (critical for drug discovery).

### Stratified Splitting
Maintain class balance in each fold for imbalanced classification problems.

## Compatibility

### Data Format
- **Input**: Compatible with existing `train_ml_model()` format
- **Output**: Similar structure to `train_ml_model()` but with list of models
- **Storage**: Uses existing resource management system (manifest + joblib)

### Model Algorithms
Supports all algorithms from `trad_ml_models.py`:
- Random Forest (Classifier/Regressor)
- Gradient Boosting (Classifier/Regressor)
- SVM (Classifier/Regressor)
- Ridge/Lasso Regression
- Logistic Regression
- K-Nearest Neighbors
- And more...

## Implementation Details

### Key Design Decisions

1. **Consistent API**: Function signature mirrors `train_ml_model()` with CV-specific parameters
2. **Reusability**: Leverages existing `get_cv_splits()` from `cross_validation.py`
3. **Flexibility**: Supports all 6 CV strategies with strategy-specific parameters
4. **Data Integrity**: Validates columns, feature vectors, and strategy requirements
5. **Progress Feedback**: Prints training progress for user visibility
6. **Error Handling**: Clear error messages for missing columns or invalid parameters

### Resource Storage Format
Models are stored using the existing `_store_resource()` system:
- **Type**: `'model'` (uses joblib for serialization)
- **Extension**: `.pkl`
- **Manifest**: Tracked in project manifest.json
- **ID**: Unique 8-character hex ID appended to filename

## Next Steps

### Potential Enhancements
1. Add evaluation during training (optional performance metrics)
2. Support for nested CV (hyperparameter tuning within CV)
3. Parallel training of models (multiprocessing)
4. Early stopping for ensemble predictions
5. Model selection based on validation performance

### Integration with Other Tools
- `predict_ml_model()`: Can be extended to support ensemble predictions
- `tune_hyperparameters()`: Can use CV models for more robust tuning
- `evaluate_ml_models()`: Can evaluate CV models across all folds

## Files Modified

1. **`src/molml_mcp/tools/ml/training.py`**
   - Implemented `train_ml_models_cv()` function (200+ lines)
   - Full docstring with all parameters documented

2. **`src/molml_mcp/tools/ml/__init__.py`**
   - Added import for `train_ml_models_cv`
   - Added to `get_all_ml_tools()` export list

3. **`tests/test_train_ml_models_cv.py`** (NEW)
   - Comprehensive test suite (6 tests, all passing)
   - Tests all major CV strategies

4. **`tests/example_train_ml_models_cv.py`** (NEW)
   - Complete examples demonstrating usage
   - Three different CV strategies shown
   - Educational comments and output

## Verification

### Function Tests
```bash
$ python -m pytest tests/test_train_ml_models_cv.py -v
================================= 6 passed in 4.35s ==================================
```

### Example Script
```bash
$ python tests/example_train_ml_models_cv.py
# Successfully demonstrates all three CV strategies
# Verifies scaffold separation, stratification, and random sampling
```

### MCP Integration
- Function automatically registered via `get_all_ml_tools()`
- Available to MCP server clients
- Compatible with existing MCP resource system

---

**Status**: ✅ Implementation complete and tested
**Ready for**: Production use and MCP server deployment
