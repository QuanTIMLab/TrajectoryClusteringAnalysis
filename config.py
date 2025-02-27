"""
Configuration settings for the visualization module.

This module defines various configuration parameters that control the behavior and appearance of the visualization. These settings include the number of patients, the maximum number of weeks, the available treatments, the sample sizes, the symbols and colors used for the glyphs, the color scale, and the size and height of the figure.

The configuration settings are stored in a dictionary named `CONFIG` that can be imported and used by other modules in the project.
"""
import plotly.express as px

CONFIG = {
    'N_PATIENTS': 200,
    'MAX_WEEK': 24,
    'TREATMENTS': ['chimio1', 'chimio2', 'radio', 'immuno'],
    'SAMPLE_SIZES': [10, 20, 50, 100],
    'SYMBOLS': {1: 'circle', 2: 'diamond', 3: 'square', 4: 'star'},
    'GLYPH_COLORS': {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'},
    'COLORSCALE': px.colors.sequential.Viridis,
    'FIGURE_HEIGHT': 5000,
    'GLYPH_BASE_SIZE': 8,
    'GLYPH_SIZE_INCREMENT': 1.5
}