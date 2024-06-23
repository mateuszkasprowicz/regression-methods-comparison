from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import split_data, impute_missing_values, encode_categories


def create_pipeline(**kwargs) -> Pipeline:

    processing_pipeline = pipeline(
        [
            node(
                func=split_data,
                inputs=["df", "params:override_me"],
                outputs=["X_train", "X_test", "y_train", " y_test"],
            ),
            node(
                func=impute_missing_values,
                inputs=["X_train", "X_test", "y_train", " y_test", "params:override_me"],
                outputs=["X_train", "X_test", "y_train", " y_test"],
            ),
            node(
                func=encode_categories,
                inputs=["X_train", "X_test", "y_train", " y_test", "params:override_me"],
                outputs=["X_train", "X_test", "y_train", " y_test"],
            ),
        ]
    )

    abalone_pipeline = pipeline(
        pipe=processing_pipeline,
        inputs={"abalone"},
        parameters={"params:override_me": "params:abalone_processing"},
        namespace="abalone",
    )

    housing_pipeline = pipeline(
        pipe=processing_pipeline,
        inputs={"housing"},
        parameters={"params:override_me": "params:housing_processing"},
        namespace="housing",
    )

    wine_quality_pipeline = pipeline(
        pipe=processing_pipeline,
        inputs={"wine_quality"},
        parameters={"params:override_me": "params:wine_quality_processing"},
        namespace="wine_quality",
    )

    return abalone_pipeline + housing_pipeline + wine_quality_pipeline
