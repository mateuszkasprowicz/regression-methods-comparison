from typing import List
import logging

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import split_data, impute_missing_values, encode_categories


TEST_TRAIN_REFS = ["X_train", "X_test", "y_train", "y_test"]


def make_suffixes(strings: List[str], suffix: str):
    return [name + suffix for name in strings]


def new_train_eval_template() -> Pipeline:

    return pipeline(
        [
            node(
                func=split_data,
                inputs=["data", "params:override_me"],
                outputs=make_suffixes(TEST_TRAIN_REFS, "_split"),
            ),
            node(
                func=impute_missing_values,
                inputs=make_suffixes(TEST_TRAIN_REFS, "_split") + ["params:override_me"],
                outputs=make_suffixes(TEST_TRAIN_REFS, "_imputed"),
            ),
            node(
                func=encode_categories,
                inputs=make_suffixes(TEST_TRAIN_REFS, "_imputed") + ["params:override_me"],
                outputs=TEST_TRAIN_REFS,
            ),
        ]
    )


def create_pipeline(datasets: List[str]) -> Pipeline:

    model_pipelines = [
        pipeline(
            pipe=new_train_eval_template(),
            parameters={"override_me": dataset},
            inputs={"data": dataset},
            namespace=dataset,
        )
        for dataset in datasets
    ]

    all_model_pipelines = sum(model_pipelines)
    output_names = all_model_pipelines.outputs()

    consolidated_model_pipelines = pipeline(
        pipe=all_model_pipelines,
        outputs=output_names,
        namespace="data_processing",
    )

    return consolidated_model_pipelines
