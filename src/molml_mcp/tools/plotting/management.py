"""
Plot management functions (remove, list).
"""

from molml_mcp.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT, _update_layout
)


def remove_plot(plot_name: str) -> dict:
    """
    Remove a plot from the visualization dashboard.
    
    Parameters
    ----------
    plot_name : str
        Name of the plot to remove (case-insensitive)
    
    Returns
    -------
    dict
        Contains removed plot name, remaining plots, and url
    
    Examples
    --------
    >>> remove_plot("MW vs LogP")
    """
    global _active_plots, _PORT
    
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    with _server_lock:
        if plot_id not in _active_plots:
            available = [_active_plots[pid]['label'] for pid in _active_plots]
            raise ValueError(
                f"Plot '{plot_name}' not found. Available plots: {available if available else 'none'}"
            )
        
        removed_label = _active_plots[plot_id]['label']
        del _active_plots[plot_id]
        
        # Update layout (no need to re-register callbacks - universal callback handles all)
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "removed_plot": removed_label,
        "remaining_plots": list(_active_plots.keys()),
        "n_remaining": len(_active_plots),
        "url": url if _active_plots else None,
        "message": f"Plot '{removed_label}' removed. {len(_active_plots)} plot(s) remaining."
    }


def list_active_plots() -> dict:
    """
    List all active plots in the visualization dashboard.
    
    Returns
    -------
    dict
        Contains plot details, url, and count
    
    Examples
    --------
    >>> list_active_plots()
    """
    global _active_plots, _PORT
    
    if not _active_plots:
        return {
            "active_plots": [],
            "n_plots": 0,
            "url": None,
            "message": "No active plots. Use plotting functions to create visualizations."
        }
    
    plots_info = []
    for plot_id, plot_data in _active_plots.items():
        plots_info.append({
            "name": plot_data['label'],
            "plot_id": plot_id,
            "type": plot_data.get('type', 'scatter'),
            "explanation": plot_data['explanation']
        })
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plots": plots_info,
        "n_plots": len(_active_plots),
        "url": url,
        "message": f"{len(_active_plots)} plot(s) active. Visit {url} to view the dashboard."
    }
