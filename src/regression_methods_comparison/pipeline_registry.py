"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from regression_methods_comparison.pipelines import data_processing as dp
from regression_methods_comparison.pipelines import ml


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    modeling_pipeline = ml.create_pipeline()
    return {
        "__default__": (data_processing_pipeline + modeling_pipeline),
        "Data processing": data_processing_pipeline,
        "Modeling": modeling_pipeline,
    }
