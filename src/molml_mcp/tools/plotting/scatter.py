"""
Scatter plot with molecular structure tooltips.
"""

from molml_mcp.infrastructure.resources import _load_resource
from molml_mcp.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT,
    _create_scatter_figure, _ensure_server_running, _update_layout
)


def add_molecular_scatter_plot(
    input_filename: str,
    x_column: str,
    y_column: str,
    project_manifest_path: str,
    plot_name: str,
    explanation: str,
    smiles_column: str = 'smiles',
    color_column: str = None,
    size_column: str = None,
    show_structures_on_hover: bool = True
) -> dict:
    """
    Add an interactive scatter plot to the persistent visualization dashboard.
    
    Creates a new tab in the Dash visualization server. If the server isn't running,
    it will be started automatically. Multiple plots can coexist as tabs.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    x_column : str
        Column name for x-axis
    y_column : str
        Column name for y-axis
    project_manifest_path : str
        Path to manifest.json
    plot_name : str
        Unique name for this plot (used as tab label and identifier)
    explanation : str
        Brief description of the plot
    smiles_column : str, default='smiles'
        Column containing SMILES strings
    color_column : str, optional
        Column to use for point colors
    size_column : str, optional
        Column to use for point sizes
    show_structures_on_hover : bool, default=True
        If True, show molecular structures on hover
    
    Returns
    -------
    dict
        Contains plot_name, url, n_molecules, x_column, y_column, active_plots
    
    Examples
    --------
    >>> add_molecular_scatter_plot(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     x_column="MW",
    ...     y_column="LogP",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="MW vs LogP",
    ...     explanation="Molecular weight vs lipophilicity"
    ... )
    """
    global _active_plots, _PORT
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    required_cols = [x_column, y_column]
    if show_structures_on_hover:
        required_cols.append(smiles_column)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) not found in dataset: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Generate unique plot ID
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    # Check if plot already exists
    if plot_id in _active_plots:
        raise ValueError(
            f"Plot '{plot_name}' already exists. Use a different name or remove it first."
        )
    
    # Create figure
    fig = _create_scatter_figure(df, x_column, y_column, color_column, size_column, plot_id)
    
    # Store plot data
    with _server_lock:
        _active_plots[plot_id] = {
            'label': plot_name,
            'dataframe': df,
            'figure': fig,
            'x_column': x_column,
            'y_column': y_column,
            'smiles_column': smiles_column,
            'color_column': color_column,
            'size_column': size_column,
            'show_structures': show_structures_on_hover,
            'explanation': explanation
        }
        
        # Ensure server is running
        _ensure_server_running()
        
        # Update layout (no need to register callbacks - already done universally)
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "url": url,
        "n_molecules": len(df),
        "x_column": x_column,
        "y_column": y_column,
        "show_structures": show_structures_on_hover,
        "active_plots": list(_active_plots.keys()),
        "message": f"Plot '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }
