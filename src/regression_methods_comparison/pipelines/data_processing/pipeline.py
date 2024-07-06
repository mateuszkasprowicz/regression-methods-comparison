from typing import List

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from regression_methods_comparison.pipelines.utils import make_suffixes, TEST_TRAIN_REFS
from .nodes import split_data, impute_missing_values, encode_categories


def new_processing_template() -> Pipeline:

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
            pipe=new_processing_template(),
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
