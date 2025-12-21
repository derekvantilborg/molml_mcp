from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def train_ml_model(
    train_input_filename: str,
    train_feature_vectors_filename: str,
    train_smiles_column: str,
    train_label_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest",
    hyperparameters: dict = None,
    task_type: str = "auto",
    random_state: int = 42
) -> dict:
    """
    Train a single machine learning model on molecular data.
    
    Args:
        train_input_filename: CSV file with SMILES and labels
        train_feature_vectors_filename: JSON file with SMILES -> feature vector mapping
        train_smiles_column: Column name for SMILES in train input file
        train_label_column: Column name for labels in train input file
        project_manifest_path: Path to project manifest.json
        output_filename: Name for output model file (without extension)
        explanation: Description of the model
        model_algorithm: ML algorithm to use (e.g., "random_forest", "gradient_boosting")
        hyperparameters: Optional dict of hyperparameters
        task_type: "classification", "regression", or "auto" (auto-detect from labels)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with model info and performance metrics
    """
    from molml_mcp.tools.ml.trad_ml_models import get_available_models
    
    # Load training data
    train_df = _load_resource(project_manifest_path, train_input_filename)
    train_features_dict = _load_resource(project_manifest_path, train_feature_vectors_filename)
    
    # Validate required columns
    if train_smiles_column not in train_df.columns:
        raise ValueError(f"SMILES column '{train_smiles_column}' not found in {train_input_filename}")
    if train_label_column not in train_df.columns:
        raise ValueError(f"Label column '{train_label_column}' not found in {train_input_filename}")
    
    # Get SMILES and labels
    train_smiles = train_df[train_smiles_column].tolist()
    train_labels = train_df[train_label_column].values
    
    # Check that all SMILES have feature vectors
    missing_train = [smi for smi in train_smiles if smi not in train_features_dict]
    if missing_train:
        raise ValueError(f"Missing feature vectors for {len(missing_train)} training SMILES. First 5: {missing_train[:5]}")
    
    # Build feature matrix
    X_train = np.array([train_features_dict[smi] for smi in train_smiles])
    y_train = train_labels
    
    # Auto-detect task type
    if task_type == "auto":
        unique_labels = np.unique(y_train)
        if len(unique_labels) <= 10 and np.all(y_train == y_train.astype(int)):
            task_type = "classification"
        else:
            task_type = "regression"
    
    # Check if model algorithm is supported
    available_models = get_available_models(task_type)
    if model_algorithm not in available_models:
        raise ValueError(f"Model '{model_algorithm}' not supported for {task_type}. Available: {available_models}")
    
    # Train the model
    model = _train_ml_model(
        X=X_train,
        y=y_train,
        model_algorithm=model_algorithm,
        hyperparameters=hyperparameters,
        task_type=task_type,
        random_state=random_state
    )
    
    # Prepare data structure matching train_ml_models_cv output format
    # Format: list of models and list of data splits (one split in this case)
    train_data_dict = {smi: label for smi, label in zip(train_smiles, y_train.tolist())}
    
    model_data = {
        "models": [model],
        "data_splits": [
            {
                "training": train_data_dict,
                "validation": {}
            }
        ],
        "model_algorithm": model_algorithm,
        "hyperparameters": hyperparameters or {},
        "task_type": task_type,
        "random_state": random_state,
        "n_features": X_train.shape[1]
    }
    
    # Store the model
    output_id = _store_resource(
        model_data,
        project_manifest_path,
        output_filename,
        explanation,
        "model"
    )
    
    return {
        "output_filename": output_id,
        "model_algorithm": model_algorithm,
        "task_type": task_type,
        "n_train_samples": len(train_smiles),
        "n_features": X_train.shape[1],
        "hyperparameters": hyperparameters or {}
    }


def train_ml_models_cv(
    input_filename: str,
    feature_columns: List[str],
    target_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest",
    hyperparameters: dict = None,
    n_models: int = 10,
    cv_strategy: str = "monte_carlo",
    train_fraction: float = 0.8,
    n_folds: int = 5,
    scaffold_column: Optional[str] = None,
    task_type: str = "auto",
    random_state: int = 42,
    store_labels: bool = True
) -> dict:
    pass


def _train_ml_model(
    X: np.ndarray,
    y: np.ndarray,
    model_algorithm: str,
    hyperparameters: dict,
    task_type: str,
    random_state: int
):
    """
    Internal function to train a machine learning model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        model_algorithm: Algorithm name (e.g., "random_forest", "gradient_boosting")
        hyperparameters: Dictionary of hyperparameters to pass to the model.
                        If None or empty, the training function will use its defaults.
        task_type: Either "classification" or "regression"
        random_state: Random seed for reproducibility
    
    Returns:
        Trained scikit-learn model object
    
    Raises:
        ValueError: If model_algorithm is not found for the given task_type
    """
    from molml_mcp.tools.ml.trad_ml_models import get_model_function
    import inspect
    
    # Get the training function for the specified model
    try:
        train_func = get_model_function(model_algorithm, task_type)
    except ValueError as e:
        raise ValueError(f"Invalid model_algorithm '{model_algorithm}' for task '{task_type}': {e}")
    
    # Prepare parameters
    params = hyperparameters.copy() if hyperparameters else {}
    
    # Check if the training function accepts random_state parameter
    sig = inspect.signature(train_func)
    if "random_state" in sig.parameters:
        params["random_state"] = random_state
    
    # Train the model
    model = train_func(X, y, **params)
    
    return model

