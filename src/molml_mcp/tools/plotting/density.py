"""
Density plot (KDE) visualization.
"""

import numpy as np
import plotly.graph_objects as go
from scipy import stats
from molml_mcp.infrastructure.resources import _load_resource
from molml_mcp.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT,
    _ensure_server_running, _update_layout
)


def add_density_plot(
    input_filename: str,
    column: str,
    project_manifest_path: str,
    plot_name: str,
    explanation: str,
    bandwidth: str = 'scott',
    fill: bool = True,
    color: str = "#577788",
    show_rug: bool = True,
    show_mean_line: bool = True,
    show_median_line: bool = False
) -> dict:
    """
    Add an interactive density plot (KDE) to the persistent visualization dashboard.
    
    Creates a new tab in the Dash visualization server showing a kernel density estimate.
    If the server isn't running, it will be started automatically.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    column : str
        Column name to plot as density
    project_manifest_path : str
        Path to manifest.json
    plot_name : str
        Unique name for this plot (used as tab label and identifier)
    explanation : str
        Brief description of the plot
    bandwidth : str, default='scott'
        Method to determine bandwidth: 'scott', 'silverman', or a numeric value
    fill : bool, default=True
        If True, fill the area under the density curve
    color : str, default="#577788"
        Color for density curve (hex color)
    show_rug : bool, default=True
        If True, show rug plot (individual data points) at bottom
    show_mean_line : bool, default=True
        If True, show a vertical line at the mean
    show_median_line : bool, default=False
        If True, show a vertical line at the median
    
    Returns
    -------
    dict
        Contains plot_name, url, n_values, statistics, active_plots
    
    Examples
    --------
    >>> add_density_plot(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     column="MW",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="Molecular Weight Density",
    ...     explanation="Density distribution of molecular weights"
    ... )
    """
    global _active_plots, _PORT
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract data and remove NaN
    data = df[column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"Column '{column}' contains no valid (non-NaN) values")
    
    if len(data) < 2:
        raise ValueError(f"Need at least 2 data points for density estimation, got {len(data)}")
    
    # Calculate statistics
    mean_val = float(data.mean())
    median_val = float(data.median())
    min_val = float(data.min())
    max_val = float(data.max())
    std_val = float(data.std())
    
    # Generate unique plot ID
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    # Check if plot already exists
    if plot_id in _active_plots:
        raise ValueError(
            f"Plot '{plot_name}' already exists. Use a different name or remove it first."
        )
    
    # Create density estimate using KDE
    data_array = np.array(data)
    
    # Determine bandwidth
    if bandwidth == 'scott':
        bw_method = 'scott'
    elif bandwidth == 'silverman':
        bw_method = 'silverman'
    else:
        try:
            bw_method = float(bandwidth)
        except (ValueError, TypeError):
            bw_method = 'scott'
    
    kde = stats.gaussian_kde(data_array, bw_method=bw_method)
    
    # Create x values for smooth curve
    x_range = max_val - min_val
    x_min = min_val - 0.1 * x_range
    x_max = max_val + 0.1 * x_range
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = kde(x_vals)
    
    # Create density plot figure
    fig = go.Figure()
    
    # Add density curve
    if fill:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Density',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color,
            opacity=0.6
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Density',
            line=dict(color=color, width=2)
        ))
    
    # Add rug plot (individual data points)
    if show_rug:
        # Sample if too many points
        rug_data = data_array if len(data_array) <= 1000 else np.random.choice(data_array, 1000, replace=False)
        rug_y = [-0.02 * max(y_vals)] * len(rug_data)  # Small negative offset
        fig.add_trace(go.Scatter(
            x=rug_data,
            y=rug_y,
            mode='markers',
            name='Data points',
            marker=dict(
                color=color,
                size=4,
                symbol='line-ns-open',
                line=dict(width=1)
            ),
            showlegend=False
        ))
    
    # Add mean line
    if show_mean_line:
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top"
        )
    
    # Add median line
    if show_median_line:
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top"
        )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title=column),
        yaxis=dict(title="Density"),
        plot_bgcolor='rgba(255,255,255,0.1)',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x'
    )
    
    # Store plot data
    with _server_lock:
        _active_plots[plot_id] = {
            'label': plot_name,
            'figure': fig,
            'type': 'density',
            'column': column,
            'show_structures': False,
            'explanation': explanation,
            'statistics': {
                'n_values': len(data),
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
        }
        
        # Ensure server is running
        _ensure_server_running()
        
        # Update layout
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "url": url,
        "n_values": len(data),
        "statistics": {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val
        },
        "active_plots": list(_active_plots.keys()),
        "message": f"Density plot '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }
