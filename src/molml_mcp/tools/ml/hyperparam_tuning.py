from molml_mcp.tools.ml.cross_validation import _cross_validate_and_eval
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def _define_search_space(param_grid, search_strategy, n_searches, random_state=42):
    from sklearn.model_selection import ParameterGrid
    import random

    if search_strategy == "grid":
        # Full grid search
        return list(ParameterGrid(param_grid))
    elif search_strategy == "random":
        # Random search - sample without replacement to ensure each parameter set is tried only once
        all_params = list(ParameterGrid(param_grid))
        if n_searches >= len(all_params):
            # If n_searches >= total combinations, just return all (equivalent to grid search)
            return all_params
        # Set seed for reproducibility
        random.seed(random_state)
        return random.sample(all_params, n_searches)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")


def tune_hyperparameters(
    input_filename: str,
    feature_vectors_filename: str,
    smiles_column: str,     
    target_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest",
    param_grid: dict = None,
    search_strategy: str = "grid",
    n_searches: int = 50,
    cv_strategy: str = "stratified",
    n_folds: int = 5,
    val_size: float = 0.2,
    scaffold_type: str = "bemis_murcko",
    shuffle: bool = True,
    p: int = 1,
    max_splits: int = 100,
    cluster_column: str = None,
    higher_is_better: bool = True,
    metric: str = "auto",
    random_state: int = 42
) -> dict:
    
    # load training data
    train_df = _load_resource(project_manifest_path, input_filename)

    # get list of dicts of hyperparams to explore from the param_grid (full grid for grid search, random with size n for random search)
    hyper_params = _define_search_space(param_grid, search_strategy, n_searches, random_state)  # list of dicts

    # perform cross-validation for each set of hyperparams and collect results
    cv_results = []
    for params in hyper_params:
        score = _cross_validate_and_eval(model_algorithm=model_algorithm,
                                         dataset=train_df,
                                         smiles_column=smiles_column,
                                         label_column=target_column,
                                         feature_vector_dict=feature_vectors_filename,
                                         cv_strategy=cv_strategy,
                                         n_folds=n_folds,
                                         random_state=random_state,
                                         metric=metric,
                                         hyperparameters=params,
                                         cluster_column=cluster_column,
                                         val_size=val_size,
                                         scaffold_type=scaffold_type,
                                         shuffle=shuffle,
                                         p=p,
                                         max_splits=max_splits)
        
        cv_results.append(score)
        
    # get the best hyperparams based on the best score index
    if higher_is_better:
        best_index = cv_results.index(max(cv_results))
    else:
        best_index = cv_results.index(min(cv_results))

    best_hyperparams = hyper_params[best_index]

    # store the best hyperparams as a json resource
    output_filename = _store_resource(best_hyperparams, project_manifest_path, output_filename, explanation, 'json')
        
    return {
        "output_filename": output_filename,
        "best_hyperparameters": best_hyperparams,
        "best_score": cv_results[best_index],
    }   


