


def train_ml_model(
    input_filename: str,
    feature_columns: List[str],
    target_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest",
    hyperparameters: dict = None,
    val_dataset: Optional[str] = None,
    task_type: str = "auto",
    random_state: int = 42,
    store_labels: bool = True
) -> dict:
    pass


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
    pass

