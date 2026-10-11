# app/analysis/__init__.py
#
# Expose the five analytical functions at the package level so callers can do:
#   from app.analysis import get_dataset_overview, filter_data, ...

from app.analysis.overview import get_dataset_overview
from app.analysis.statistics import get_summary_statistics
from app.analysis.filtering import filter_data
from app.analysis.correlations import analyze_correlations
from app.analysis.visualization import generate_visualization
from app.analysis.aggregation import group_and_aggregate

__all__ = [
    "get_dataset_overview",
    "get_summary_statistics",
    "filter_data",
    "analyze_correlations",
    "generate_visualization",
    "group_and_aggregate",
]
