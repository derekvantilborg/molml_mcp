import numpy as np
from typing import Dict, Any


def _eval_single_ml_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    task_type: str
) -> float:
    """
    Evaluate a single ML model on a dataset using a specific metric.
    
    Args:
        model: Trained scikit-learn model
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        metric: Metric name (e.g., "accuracy", "f1_score", "mse", "r2")
        task_type: Either "classification" or "regression"
    
    Returns:
        Metric value as float (or None if metric cannot be computed)
    """
    from molml_mcp.tools.ml.metrics import get_metric_function
    
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
    metric_func = get_metric_function(metric)
    return metric_func(y, y_pred)
