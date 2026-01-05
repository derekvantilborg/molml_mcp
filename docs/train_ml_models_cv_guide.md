# train_ml_models_cv Quick Reference

## Basic Usage

```python
from molml_mcp.tools.ml.training import train_ml_models_cv

result = train_ml_models_cv(
    input_filename="dataset.csv",               # CSV with SMILES and labels
    feature_vectors_filename="features.json",   # SMILES -> feature mapping
    smiles_column="smiles",
    label_column="label",
    project_manifest_path="path/to/manifest.json",
    output_filename="my_cv_models",
    explanation="Cross-validated models for prediction",
    cv_strategy="montecarlo",                   # CV strategy
    n_folds=5,                                  # Number of models
    random_state=42
)
```

## Cross-Validation Strategies

### 1. Monte Carlo (Random Sampling)
**Use when**: You want flexible validation size and repeated random sampling

```python
result = train_ml_models_cv(
    ...,
    cv_strategy="montecarlo",
    n_folds=10,              # Number of random splits
    val_size=0.2,            # 20% validation
    random_state=42
)
```

### 2. K-Fold
**Use when**: You want exhaustive CV with equal-sized folds

```python
result = train_ml_models_cv(
    ...,
    cv_strategy="kfold",
    n_folds=5,               # 5-fold CV
    shuffle=True,            # Shuffle before splitting
    random_state=42
)
```

### 3. Stratified K-Fold
**Use when**: You have imbalanced classes and want to maintain class distribution

```python
result = train_ml_models_cv(
    ...,
    cv_strategy="stratified",
    n_folds=5,
    random_state=42
)
# Automatically bins continuous labels for stratification
```

### 4. Scaffold-Based
**Use when**: Testing generalization to new chemical scaffolds (drug discovery)

```python
# First, compute scaffolds and add to dataset
df['scaffold'] = df['smiles'].apply(get_murcko_scaffold)

result = train_ml_models_cv(
    ...,
    cv_strategy="scaffold",
    scaffold_column="scaffold",  # Column with scaffold SMILES
    n_folds=5,
    random_state=42
)
# Molecules with same scaffold stay together in train or validation
```

### 5. Cluster-Based
**Use when**: You have pre-computed molecular clusters

```python
# First, compute clusters and add to dataset
df['cluster'] = compute_clusters(df['smiles'])

result = train_ml_models_cv(
    ...,
    cv_strategy="cluster",
    cluster_column="cluster",    # Column with cluster IDs
    n_folds=5,
    random_state=42
)
# All molecules in same cluster stay together
```

### 6. Leave-P-Out
**Use when**: You have small dataset and want to test on every P-size subset

```python
result = train_ml_models_cv(
    ...,
    cv_strategy="leavepout",
    p=2,                     # Leave 2 samples out
    max_splits=50,           # Limit splits (combinatorial explosion!)
    random_state=42
)
# WARNING: Can generate C(n, p) splits - use max_splits!
```

## Hyperparameter Configuration

```python
# Random Forest Classifier
result = train_ml_models_cv(
    ...,
    model_algorithm="random_forest_classifier",
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    }
)

# Support Vector Regressor
result = train_ml_models_cv(
    ...,
    model_algorithm="svr",
    hyperparameters={
        'C': 1.0,
        'epsilon': 0.1,
        'kernel': 'rbf'
    }
)

# Ridge Regression
result = train_ml_models_cv(
    ...,
    model_algorithm="ridge",
    hyperparameters={
        'alpha': 1.0
    }
)
```

## Return Value

```python
{
    'output_filename': 'my_cv_models_A3F2B1D4.pkl',  # Saved model resource
    'model_algorithm': 'random_forest_classifier',
    'n_models': 5,                                   # Number of models trained
    'n_features': 2048,                              # Features per model
    'cv_strategy': 'montecarlo',
    'n_folds': 5,
    'hyperparameters': {...}
}
```

## Stored Model Structure

```python
from molml_mcp.infrastructure.resources import _load_resource

model_data = _load_resource(manifest_path, result['output_filename'])

# Structure:
{
    'models': [model1, model2, ...],           # Trained scikit-learn models
    'data_splits': [
        {
            'training': {'SMILES': label, ...},
            'validation': {'SMILES': label, ...}
        },
        ...
    ],
    'model_algorithm': 'random_forest_classifier',
    'hyperparameters': {...},
    'n_features': 2048,
    'cv_strategy': 'montecarlo',
    'cv_parameters': {...}
}
```

## Using the Models

### Ensemble Predictions
```python
import numpy as np

# Load models
model_data = _load_resource(manifest_path, model_filename)

# Make predictions with all models
predictions = []
for model in model_data['models']:
    pred = model.predict(X_test)
    predictions.append(pred)

# Average predictions (ensemble)
ensemble_pred = np.mean(predictions, axis=0)
```

### Cross-Validation Evaluation
```python
from molml_mcp.tools.ml.evaluation import _eval_single_ml_model

# Evaluate each model on its validation set
scores = []
for model, split in zip(model_data['models'], model_data['data_splits']):
    # Get validation data
    val_smiles = list(split['validation'].keys())
    val_labels = list(split['validation'].values())
    
    # Get feature vectors
    X_val = np.array([feature_vectors[smi] for smi in val_smiles])
    y_val = np.array(val_labels)
    
    # Evaluate
    score = _eval_single_ml_model(model, X_val, y_val, metric='r2')
    scores.append(score)

# Cross-validation score
mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"CV Score: {mean_score:.3f} ± {std_score:.3f}")
```

## Common Patterns

### Pattern 1: Robust Model Training
```python
# Train multiple models for robust predictions
result = train_ml_models_cv(
    input_filename=data_file,
    feature_vectors_filename=features_file,
    smiles_column='smiles',
    label_column='activity',
    project_manifest_path=manifest_path,
    output_filename='robust_models',
    explanation='Multiple models for ensemble prediction',
    model_algorithm='random_forest_regressor',
    cv_strategy='montecarlo',
    n_folds=10,
    val_size=0.2,
    random_state=42
)
```

### Pattern 2: Scaffold Generalization Testing
```python
# Test if model generalizes to new scaffolds
result = train_ml_models_cv(
    ...,
    cv_strategy='scaffold',
    scaffold_column='murcko_scaffold',
    n_folds=5,
    explanation='Test scaffold generalization'
)
```

### Pattern 3: Imbalanced Classification
```python
# Handle imbalanced classes with stratification
result = train_ml_models_cv(
    ...,
    cv_strategy='stratified',
    n_folds=5,
    explanation='Stratified CV for imbalanced classes'
)
```

## Tips and Best Practices

### Choosing n_folds
- **Monte Carlo**: 5-20 splits (more = more robust estimates)
- **K-Fold**: 5-10 folds (more = less data per fold)
- **Scaffold**: Limited by number of unique scaffolds
- **Leave-P-Out**: Use `max_splits` to avoid explosion

### Validation Size
- **Small datasets** (n < 100): 10-20% validation
- **Medium datasets** (100-1000): 15-25% validation
- **Large datasets** (> 1000): 20-30% validation

### Random State
Always set `random_state` for reproducibility!

### Model Selection
- **Random Forest**: Good default, handles non-linearity
- **SVM**: Good for small-medium datasets
- **Ridge/Lasso**: Good for linear relationships
- **Gradient Boosting**: Best performance but slower

## Error Handling

```python
# Missing column
try:
    result = train_ml_models_cv(...)
except ValueError as e:
    print(f"Column error: {e}")

# Missing feature vectors
try:
    result = train_ml_models_cv(...)
except ValueError as e:
    print(f"Feature error: {e}")

# Invalid CV strategy
try:
    result = train_ml_models_cv(..., cv_strategy='invalid')
except ValueError as e:
    print(f"CV strategy error: {e}")
```

## Integration with Other Tools

### With tune_hyperparameters()
```python
# 1. Find best hyperparameters
best_params = tune_hyperparameters(...)

# 2. Train multiple models with best params
result = train_ml_models_cv(
    ...,
    hyperparameters=best_params,
    cv_strategy='montecarlo',
    n_folds=10
)
```

### With predict_ml_model()
```python
# Train CV models
cv_result = train_ml_models_cv(...)

# Load and use for predictions
model_data = _load_resource(manifest_path, cv_result['output_filename'])

# Use first model for quick predictions
predictions = model_data['models'][0].predict(X_new)

# Or ensemble predictions
ensemble = np.mean([m.predict(X_new) for m in model_data['models']], axis=0)
```

## See Also
- `train_ml_model()`: Train single model (no CV)
- `tune_hyperparameters()`: Optimize hyperparameters
- `predict_ml_model()`: Make predictions
- `calculate_metrics()`: Evaluate model performance
