"""
Plotting tools for molecular data visualization.
"""

from molml_mcp.tools.plotting.scatter import add_molecular_scatter_plot
from molml_mcp.tools.plotting.histogram import add_histogram
from molml_mcp.tools.plotting.density import add_density_plot
from molml_mcp.tools.plotting.box import add_box_plot
from molml_mcp.tools.plotting.heatmap import add_correlation_heatmap, add_grouped_heatmap
from molml_mcp.tools.plotting.management import remove_plot, list_active_plots

__all__ = [
    'add_molecular_scatter_plot',
    'add_histogram',
    'add_density_plot',
    'add_box_plot',
    'add_correlation_heatmap',
    'add_grouped_heatmap',
    'remove_plot',
    'list_active_plots',
]


def get_all_plotting_tools():
    """Return list of all plotting tool functions."""
    return [
        add_molecular_scatter_plot,
        add_histogram,
        add_density_plot,
        add_box_plot,
        add_correlation_heatmap,
        add_grouped_heatmap,
        remove_plot,
        list_active_plots,
    ]
