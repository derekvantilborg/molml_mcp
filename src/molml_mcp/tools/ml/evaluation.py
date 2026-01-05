import numpy as np
from typing import Dict, Any, List, Optional


def _eval_single_ml_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metric: str
) -> float:
    """
    Evaluate a single ML model on a dataset using a specific metric.
    
    Args:
        model: Trained scikit-learn model
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        metric: Metric name (e.g., "accuracy", "f1_score", "mse", "r2")
    
    Returns:
        Metric value as float (or None if metric cannot be computed)
    """
    from molml_mcp.tools.ml.metrics import _get_metric_function
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Special handling for ROC AUC (requires probabilities)
    if metric == "roc_auc":
        if not hasattr(model, "predict_proba"):
            return None
        try:
            y_pred = model.predict_proba(X)[:, 1]
        except:
            return None
    
    # Get metric function and compute
    metric_func = _get_metric_function(metric)
    return metric_func(y, y_pred)


def predict_ml_model(
    ml_model_filename: str,
    test_input_filename: str,
    test_feature_vectors_filename: str,
    test_smiles_column: str,
    predict_column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
) -> dict:
    """
    Predict using a trained ML model on a test dataset.
    
    This function loads a trained model and applies it to a test dataset to generate
    predictions. The predictions are added as a new column to the test dataset and
    stored as a new resource.
    
    After making predictions, you can evaluate model performance by using the
    calculate_metrics() function from metrics.py to compare predictions against
    true labels (if available in the test dataset).
    
    Args:
        ml_model_filename: Filename of the trained model to use for predictions
        test_input_filename: Filename of the test dataset (CSV with SMILES)
        test_feature_vectors_filename: Filename of the feature vectors (JSON dict {smiles: [features]})
        test_smiles_column: Name of the SMILES column in the test dataset
        predict_column_name: Name for the new prediction column (e.g., "predicted_label")
        project_manifest_path: Path to manifest.json
        output_filename: Name for the output dataset with predictions
        explanation: Description of this prediction operation
    
    Returns:
        dict with output_filename, n_predictions, columns, and preview
        
    Example:
        >>> # Make predictions
        >>> result = predict_ml_model(
        ...     ml_model_filename="random_forest_A1B2C3D4.pkl",
        ...     test_input_filename="test_data_E5F6G7H8.csv",
        ...     test_feature_vectors_filename="test_features_I9J0K1L2.json",
        ...     test_smiles_column="smiles",
        ...     predict_column_name="predicted_activity",
        ...     project_manifest_path="/path/to/manifest.json",
        ...     output_filename="predictions",
        ...     explanation="Predictions on test set using Random Forest model"
        ... )
        >>> 
        >>> # Evaluate predictions (if test data has true labels)
        >>> from molml_mcp.tools.ml.metrics import calculate_metrics
        >>> metrics = calculate_metrics(
        ...     input_filename=result['output_filename'],
        ...     project_manifest_path="/path/to/manifest.json",
        ...     true_label_column="true_activity",
        ...     predicted_column="predicted_activity",
        ...     metrics=["accuracy", "precision", "recall", "f1_score"]
        ... )
    """
    from molml_mcp.infrastructure.resources import _load_resource, _store_resource
    import pandas as pd
    import numpy as np
    
    # Load model data (could be a dict structure or raw model for backwards compatibility)
    model_data = _load_resource(project_manifest_path, ml_model_filename)
    
    # Extract the actual model from the structure
    if isinstance(model_data, dict) and "models" in model_data:
        # New format from train_ml_model: {"models": [model], "data_splits": [...], ...}
        model = model_data["models"][0]
    else:
        # Backwards compatibility: assume it's the model directly
        model = model_data
    
    # Load test dataset
    test_df = _load_resource(project_manifest_path, test_input_filename)
    
    # Load feature vectors
    feature_vectors = _load_resource(project_manifest_path, test_feature_vectors_filename)
    
    # Validate SMILES column exists
    if test_smiles_column not in test_df.columns:
        raise ValueError(f"SMILES column '{test_smiles_column}' not found in test dataset")
    
    # Extract SMILES from test dataset
    test_smiles = test_df[test_smiles_column].tolist()
    
    # Build feature matrix (ensure order matches test_smiles)
    X_test = np.array([feature_vectors[smi] for smi in test_smiles])
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Add predictions to dataset
    output_df = test_df.copy()
    output_df[predict_column_name] = predictions
    
    # Store output dataset
    output_id = _store_resource(
        output_df,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_predictions": len(predictions),
        "columns": output_df.columns.tolist(),
        "preview": output_df.head(5).to_dict('records')
    }


def _evaluate_fold_metrics(model, data_dict, feature_vectors, metrics, fold_idx):
    """Helper to evaluate metrics for a single fold."""
    if not data_dict:
        return None
    
    smiles_list = list(data_dict.keys())
    labels = np.array(list(data_dict.values()))
    
    missing = [s for s in smiles_list if s not in feature_vectors]
    if missing:
        raise ValueError(f"Fold {fold_idx + 1}: Missing {len(missing)} feature vectors")
    
    X = np.array([feature_vectors[s] for s in smiles_list])
    fold_metrics = {"fold": fold_idx + 1, "n_samples": len(smiles_list), "metrics": {}}
    
    for metric_name in metrics:
        try:
            fold_metrics["metrics"][metric_name] = _eval_single_ml_model(model, X, labels, metric_name)
        except Exception:
            fold_metrics["metrics"][metric_name] = None
    
    return fold_metrics


def _aggregate_metrics(per_fold_metrics, metrics):
    """Helper to aggregate metrics across folds."""
    summary = {}
    for metric_name in metrics:
        values = [f["metrics"][metric_name] for f in per_fold_metrics 
                  if f["metrics"].get(metric_name) is not None]
        
        if values:
            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_folds": len(values),
                "values": [float(v) for v in values]
            }
        else:
            summary[metric_name] = {
                "mean": None, "std": None, "min": None, "max": None,
                "n_folds": 0, "values": []
            }
    return summary


def evaluate_cv_models(
    cv_model_filename: str,
    feature_vectors_filename: str,
    project_manifest_path: str,
    metrics: List[str],
    output_filename: str,
    explanation: str = "Cross-validation evaluation results",
    use_validation_sets: bool = True,
    evaluate_training_sets: bool = False,
    test_input_filename: Optional[str] = None,
    test_feature_vectors_filename: Optional[str] = None,
    test_smiles_column: Optional[str] = None,
    test_label_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate cross-validated models trained with train_ml_models_cv().
    
    Supports validation set evaluation (CV performance), training set evaluation 
    (overfitting check), and test set evaluation (ensemble consistency).
    
    Args:
        cv_model_filename: Filename of CV models from train_ml_models_cv()
        feature_vectors_filename: JSON file with SMILES -> feature vector mapping
        project_manifest_path: Path to manifest.json
        metrics: List of metric names (e.g., ["accuracy", "f1_score", "r2"])
        output_filename: Name for output JSON report
        explanation: Description of this evaluation
        use_validation_sets: If True, evaluate on validation sets (default)
        evaluate_training_sets: If True, also evaluate on training sets
        test_input_filename: CSV file with test data (for test set evaluation)
        test_feature_vectors_filename: Optional separate test feature vectors
        test_smiles_column: SMILES column name in test data
        test_label_column: Label column name in test data
        
    Returns:
        Dict with output_filename, n_models, cv_strategy, metrics_summary, 
        training_metrics_summary (if applicable), and n_folds_evaluated
    """
    from molml_mcp.infrastructure.resources import _load_resource, _store_resource
    import pandas as pd
    
    # Load and validate CV model data
    model_data = _load_resource(project_manifest_path, cv_model_filename)
    if not isinstance(model_data, dict) or "models" not in model_data:
        raise ValueError("Invalid CV model file. Expected dict with 'models' key")
    
    models = model_data["models"]
    data_splits = model_data.get("data_splits", [])
    if not models:
        raise ValueError("No models found in CV model file")
    
    feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
    evaluation_mode = "validation" if use_validation_sets else "test"
    per_fold_metrics = []
    per_fold_training_metrics = [] if evaluate_training_sets else None
    
    if use_validation_sets:
        if len(data_splits) != len(models):
            raise ValueError(f"Model count ({len(models)}) doesn't match data splits ({len(data_splits)})")
        
        for fold_idx, (model, split) in enumerate(zip(models, data_splits)):
            # Validation metrics
            fold_result = _evaluate_fold_metrics(
                model, split.get("validation", {}), feature_vectors, metrics, fold_idx
            )
            if fold_result:
                per_fold_metrics.append(fold_result)
            
            # Training metrics (if requested)
            if evaluate_training_sets:
                train_result = _evaluate_fold_metrics(
                    model, split.get("training", {}), feature_vectors, metrics, fold_idx
                )
                if train_result:
                    per_fold_training_metrics.append(train_result)
    
    else:
        # Test set evaluation
        if not all([test_input_filename, test_smiles_column, test_label_column]):
            raise ValueError("test_input_filename, test_smiles_column, and test_label_column required")
        
        test_df = _load_resource(project_manifest_path, test_input_filename)
        if test_smiles_column not in test_df.columns or test_label_column not in test_df.columns:
            raise ValueError(f"Required columns not found in test data")
        
        test_smiles = test_df[test_smiles_column].tolist()
        test_labels = test_df[test_label_column].values
        
        test_features = (_load_resource(project_manifest_path, test_feature_vectors_filename) 
                        if test_feature_vectors_filename else feature_vectors)
        
        missing = [s for s in test_smiles if s not in test_features]
        if missing:
            raise ValueError(f"Missing {len(missing)} test feature vectors")
        
        X_test = np.array([test_features[s] for s in test_smiles])
        
        for fold_idx, model in enumerate(models):
            fold_metrics = {"fold": fold_idx + 1, "n_samples": len(test_smiles), "metrics": {}}
            for metric_name in metrics:
                try:
                    fold_metrics["metrics"][metric_name] = _eval_single_ml_model(
                        model, X_test, test_labels, metric_name
                    )
                except Exception:
                    fold_metrics["metrics"][metric_name] = None
            per_fold_metrics.append(fold_metrics)
    
    # Aggregate metrics
    metrics_summary = _aggregate_metrics(per_fold_metrics, metrics)
    training_metrics_summary = (_aggregate_metrics(per_fold_training_metrics, metrics) 
                                if evaluate_training_sets and per_fold_training_metrics else None)
    
    # Build report
    report = {
        "evaluation_type": "cross_validation_evaluation",
        "cv_model_filename": cv_model_filename,
        "feature_vectors_filename": feature_vectors_filename,
        "evaluation_mode": evaluation_mode,
        "n_models": len(models),
        "model_algorithm": model_data.get("model_algorithm", "unknown"),
        "cv_strategy": model_data.get("cv_strategy", "unknown"),
        "cv_parameters": model_data.get("cv_parameters", {}),
        "metrics_requested": metrics,
        "metrics_summary": metrics_summary,
        "per_fold_metrics": per_fold_metrics
    }
    
    if training_metrics_summary:
        report["training_metrics_summary"] = training_metrics_summary
        report["per_fold_training_metrics"] = per_fold_training_metrics
    
    if not use_validation_sets:
        report["test_dataset"] = {
            "filename": test_input_filename,
            "n_samples": len(test_smiles),
            "smiles_column": test_smiles_column,
            "label_column": test_label_column,
            "feature_vectors_filename": test_feature_vectors_filename or feature_vectors_filename
        }
    
    output_id = _store_resource(report, project_manifest_path, output_filename, explanation, "json")
    
    return {
        "output_filename": output_id,
        "n_models": len(models),
        "cv_strategy": model_data.get("cv_strategy", "unknown"),
        "evaluation_mode": evaluation_mode,
        "metrics_summary": metrics_summary,
        "training_metrics_summary": training_metrics_summary,
        "n_folds_evaluated": len(per_fold_metrics)
    }
