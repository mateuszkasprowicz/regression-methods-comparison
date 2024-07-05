"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from regression_methods_comparison.pipelines import data_processing as dp
from regression_methods_comparison.pipelines import ml


CONF_PATH = str(settings.CONF_SOURCE)
CONF_LOADER = OmegaConfigLoader(conf_source=CONF_PATH)
PARAMETERS = CONF_LOADER["parameters"]

DATASETS = PARAMETERS["datasets"]
MODEL_TYPES = PARAMETERS["models"]


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline(datasets=DATASETS)
    ml_pipeline = ml.create_pipeline(datasets=DATASETS, model_types=MODEL_TYPES)
    return {
        "__default__": (data_processing_pipeline + ml_pipeline),
        "data_processing": data_processing_pipeline,
        "ml": ml_pipeline,
    }
