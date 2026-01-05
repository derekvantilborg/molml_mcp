# evaluate_cv_models Implementation Summary

## Overview
Implemented `evaluate_cv_models()` function in `src/molml_mcp/tools/ml/evaluation.py` to comprehensively evaluate cross-validated models trained with `train_ml_models_cv()`.

## Key Features

### Two Evaluation Modes

1. **Validation Set Evaluation** (default):
   - Each model evaluated on its held-out validation set from CV split
   - Provides true cross-validation performance estimate
   - Assesses model generalization across different data subsets

2. **Test Set Evaluation**:
   - All models evaluated on independent test set
   - Assesses model consistency and ensemble potential
   - Useful for comparing CV training strategies

### Function Signature
```python
def evaluate_cv_models(
    cv_model_filename: str,                    # CV models from train_ml_models_cv()
    feature_vectors_filename: str,             # SMILES -> feature mapping
    project_manifest_path: str,                # Path to manifest.json
    metrics: List[str],                        # Metrics to compute
    output_filename: str,                      # Output report name
    explanation: str = "...",                  # Description
    use_validation_sets: bool = True,          # Eval mode
    test_input_filename: Optional[str] = None, # For test mode
    test_smiles_column: Optional[str] = None,  # For test mode
    test_label_column: Optional[str] = None    # For test mode
) -> Dict[str, Any]
```

### Computed Statistics
For each metric across all CV folds:
- **Mean**: Average performance
- **Std**: Performance variability (model consistency)
- **Min/Max**: Best and worst fold performance
- **Per-fold values**: Individual fold metrics
- **N folds**: Number of successful evaluations

### Output Structure

**Returned Dictionary:**
```python
{
    "output_filename": str,           # Saved JSON report
    "n_models": int,                  # Number of models evaluated
    "cv_strategy": str,               # CV strategy used
    "evaluation_mode": str,           # 'validation' or 'test'
    "metrics_summary": {              # Aggregated statistics
        "metric_name": {
            "mean": float,
            "std": float,
            "min": float,
            "max": float,
            "n_folds": int,
            "values": List[float]
        }
    },
    "n_folds_evaluated": int
}
```

**Saved JSON Report:**
```python
{
    "evaluation_type": "cross_validation_evaluation",
    "timestamp": str,
    "cv_model_filename": str,
    "feature_vectors_filename": str,
    "evaluation_mode": "validation" | "test",
    "n_models": int,
    "model_algorithm": str,
    "cv_strategy": str,
    "cv_parameters": dict,
    "metrics_requested": List[str],
    "metrics_summary": dict,          # Aggregated stats
    "per_fold_metrics": [             # Detailed per-fold results
        {
            "fold": int,
            "n_samples": int,
            "metrics": {
                "metric_name": float
            }
        }
    ],
    "test_dataset": dict              # Only if test mode
}
```

## Supported Metrics

### Classification Metrics
- `accuracy`: Overall accuracy
- `balanced_accuracy`: Balanced accuracy (for imbalanced data)
- `precision`: Precision score
- `recall`: Recall/sensitivity
- `f1_score`: F1 score (harmonic mean of precision/recall)
- `roc_auc`: ROC AUC score (requires probabilities)
- `matthews_corrcoef`: Matthews correlation coefficient

### Regression Metrics
- `mse`: Mean squared error
- `mae`: Mean absolute error
- `r2`: R-squared score
- `rmse`: Root mean squared error

## Use Cases

### 1. Cross-Validation Performance Assessment
Evaluate each model on its validation set to get robust performance estimates:
```python
result = evaluate_cv_models(
    cv_model_filename="cv_models.pkl",
    feature_vectors_filename="features.json",
    project_manifest_path="manifest.json",
    metrics=["accuracy", "f1_score"],
    output_filename="cv_evaluation",
    use_validation_sets=True  # CV mode
)

print(f"CV Accuracy: {result['metrics_summary']['accuracy']['mean']:.3f} "
      f"± {result['metrics_summary']['accuracy']['std']:.3f}")
```

### 2. Model Consistency Assessment
Evaluate all models on same test set to check consistency:
```python
result = evaluate_cv_models(
    cv_model_filename="cv_models.pkl",
    feature_vectors_filename="features.json",
    project_manifest_path="manifest.json",
    metrics=["r2", "mse"],
    output_filename="test_evaluation",
    use_validation_sets=False,  # Test mode
    test_input_filename="test.csv",
    test_smiles_column="smiles",
    test_label_column="activity"
)

# Low std = consistent models = good for ensemble
std_r2 = result['metrics_summary']['r2']['std']
print(f"Model consistency (R² std): {std_r2:.4f}")
```

### 3. CV Strategy Comparison
Compare different CV strategies by evaluating their models:
```python
strategies = ['kfold', 'stratified', 'montecarlo']
results = {}

for strategy in strategies:
    # Train models with strategy
    train_result = train_ml_models_cv(..., cv_strategy=strategy)
    
    # Evaluate
    eval_result = evaluate_cv_models(
        cv_model_filename=train_result['output_filename'],
        ...
    )
    
    results[strategy] = eval_result['metrics_summary']

# Compare mean and std across strategies
```

### 4. Model Selection
Identify best and worst performing folds:
```python
result = evaluate_cv_models(...)
report = _load_resource(manifest, result['output_filename'])

# Find best fold
best_fold = max(
    report['per_fold_metrics'],
    key=lambda x: x['metrics']['accuracy']
)
print(f"Best fold: {best_fold['fold']} with {best_fold['metrics']['accuracy']:.3f}")
```

### 5. Ensemble Validation
Check if ensemble will be beneficial:
```python
result = evaluate_cv_models(..., use_validation_sets=False, ...)

# Low variance = models agree = ensemble may not help much
# High variance = models disagree = ensemble could improve
variance = result['metrics_summary']['r2']['std']
if variance < 0.05:
    print("Models are very consistent - ensemble may not add much value")
else:
    print("Models show diversity - ensemble recommended")
```

## Implementation Details

### Progress Feedback
The function prints progress during evaluation:
```
Evaluating 5 CV models in 'validation' mode...
Metrics: accuracy, precision, recall, f1_score
  Evaluating fold 1/5...
  Evaluating fold 2/5...
  ...
Aggregating metrics across folds...
```

### Error Handling
- Validates CV model structure
- Checks for missing feature vectors
- Handles missing metrics gracefully (sets to None)
- Validates test set parameters in test mode
- Clear error messages for common issues

### Metric Computation
- Uses `_eval_single_ml_model()` for each fold
- Automatically handles classification vs regression
- Skips None values when computing statistics
- Preserves all individual fold values for analysis

### Output Format
- Console summary with formatted tables
- JSON report for programmatic access
- Resource management via manifest system
- Timestamp for tracking evaluations

## Integration

### Module Exports
Updated `src/molml_mcp/tools/ml/__init__.py`:
- Added `evaluate_cv_models` to imports
- Added to `get_all_ml_tools()` return list
- Function automatically registered with MCP server

### Workflow Integration
```
train_ml_models_cv() → evaluate_cv_models() → Report/Analysis
         ↓                       ↓
    CV Models              Performance Stats
```

Can be chained with:
- `train_ml_models_cv()`: Create models to evaluate
- `calculate_metrics()`: Additional metric computation
- `predict_ml_model()`: Make ensemble predictions

## Testing

### Test Coverage
Created comprehensive test suite in `tests/test_evaluate_cv_models.py`:

1. ✅ `test_evaluate_cv_models_validation_mode` - Validation set evaluation
2. ✅ `test_evaluate_cv_models_test_mode` - Test set evaluation
3. ✅ `test_evaluate_cv_models_regression` - Regression metrics
4. ✅ `test_evaluate_cv_models_stratified` - Stratified CV models
5. ✅ `test_evaluate_cv_models_metrics_consistency` - Metric consistency
6. ✅ `test_evaluate_cv_models_error_handling` - Error cases
7. ✅ `test_evaluate_cv_models_summary_statistics` - Stats computation

**All 7 tests pass successfully** (7.29s execution time)

### Example Script
Created `tests/example_evaluate_cv_models.py` with three examples:
1. Validation set evaluation (CV performance)
2. Test set evaluation (model consistency)
3. Comparing CV strategies

## Performance Considerations

### Efficiency
- Vectorized operations via NumPy
- Minimal data copying
- Efficient metric computation
- Progress feedback for long evaluations

### Memory
- Loads all models into memory (inherent to CV evaluation)
- Efficient feature matrix construction
- Minimal intermediate storage

### Scalability
- Handles arbitrary number of folds
- Works with any number of metrics
- Supports large test sets
- Resource-efficient for typical CV sizes (5-20 folds)

## Interpretation Guide

### Mean Values
- Average performance across all CV folds
- Best estimate of model generalization
- Use for reporting overall performance

### Standard Deviation
- **Low std (< 0.05)**: Consistent performance, stable model
- **Medium std (0.05-0.15)**: Typical variation, acceptable
- **High std (> 0.15)**: Unstable performance, investigate data issues

### Min/Max Range
- Shows performance spread
- Large range may indicate:
  - Data heterogeneity
  - Problematic folds
  - CV strategy mismatch

### Per-Fold Analysis
- Identify outlier folds
- Debug data quality issues
- Understand model behavior on different subsets

## Example Outputs

### Console Summary
```
================================================================================
CROSS-VALIDATION EVALUATION SUMMARY
================================================================================
Models evaluated: 5
CV strategy: montecarlo
Evaluation mode: validation

accuracy            : 0.8800 ± 0.0980 (n=5)
precision           : 0.9500 ± 0.1000 (n=5)
recall              : 0.8833 ± 0.1453 (n=5)
f1_score            : 0.9029 ± 0.0820 (n=5)
================================================================================

Evaluation report saved: cv_evaluation_A1B2C3D4.json
```

### Typical Results
**Good Performance:**
```
accuracy: 0.92 ± 0.03  # High mean, low variance
```

**Inconsistent Performance:**
```
accuracy: 0.75 ± 0.18  # Lower mean, high variance
```

**Overfitting Warning:**
```
Validation: r2 = 0.95 ± 0.02  # Great!
Test:       r2 = 0.65 ± 0.15  # Much worse on test set
```

## Files Modified/Created

1. **`src/molml_mcp/tools/ml/evaluation.py`**
   - Added `evaluate_cv_models()` function (~300 lines)
   - Updated imports for Optional and List types
   - Comprehensive docstring with examples

2. **`src/molml_mcp/tools/ml/__init__.py`**
   - Added `evaluate_cv_models` to imports
   - Added to `get_all_ml_tools()` exports

3. **`tests/test_evaluate_cv_models.py`** (NEW)
   - 7 comprehensive tests covering all modes
   - Tests validation/test modes, error handling
   - Tests with classification and regression

4. **`tests/example_evaluate_cv_models.py`** (NEW)
   - 3 detailed examples with explanations
   - Demonstrates both evaluation modes
   - Shows CV strategy comparison

## Verification

### Function Tests
```bash
$ python -m pytest tests/test_evaluate_cv_models.py -v
======================================= 7 passed in 7.29s =======================================
```

### Example Script
```bash
$ python tests/example_evaluate_cv_models.py
# Successfully demonstrates all evaluation modes
# Shows proper output formatting
# Validates metrics computation
```

### MCP Integration
- ✅ Function registered via `get_all_ml_tools()`
- ✅ Available to MCP server clients
- ✅ Compatible with MCP resource system
- ✅ No errors in codebase

## Best Practices

### Choosing Evaluation Mode

**Use Validation Set Evaluation when:**
- Assessing CV performance (standard practice)
- Comparing hyperparameters
- Reporting cross-validation results
- No independent test set available

**Use Test Set Evaluation when:**
- Checking model consistency
- Evaluating ensemble potential
- Final model assessment
- Independent test set available

### Metric Selection

**For Classification:**
```python
metrics = [
    'accuracy',           # Overall performance
    'balanced_accuracy',  # For imbalanced data
    'f1_score',          # Balanced metric
    'precision',         # When false positives costly
    'recall'             # When false negatives costly
]
```

**For Regression:**
```python
metrics = [
    'r2',   # Explained variance
    'mse',  # Squared error penalty
    'mae'   # Absolute error
]
```

### Interpreting Results

1. **Check mean values**: Are they acceptable for your problem?
2. **Check std values**: Is performance consistent?
3. **Check min/max**: Are there problematic outliers?
4. **Investigate outliers**: Why did certain folds perform poorly?
5. **Compare to baseline**: Is model better than naive baseline?

---

**Status**: ✅ Implementation complete and tested
**Ready for**: Production use and MCP server deployment
**Next Steps**: Can be extended with ensemble prediction functionality
