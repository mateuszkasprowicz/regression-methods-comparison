from typing import List

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from src.regression_methods_comparison.pipelines.utils import TEST_TRAIN_REFS
from .nodes import evaluate_model, train_model


def new_train_eval_template() -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:override_me"],
                outputs=["regressor", "experiment_params"],
                tags="train",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                tags="evaluate",
            ),
        ]
    )


def create_pipeline(datasets: List[str], model_types: List[str]) -> Pipeline:

    dataset_pipelines = []

    for dataset in datasets:
        model_pipelines = [
            pipeline(
                pipe=new_train_eval_template(),
                inputs={k: k for k in TEST_TRAIN_REFS},
                parameters={"override_me": model_type},
                namespace=model_type,
            )
            for model_type in model_types
        ]

        all_models_pipeline = pipeline(pipe=sum(model_pipelines), namespace=dataset)
        dataset_pipelines.append(all_models_pipeline)

    all_dataset_pipelines = sum(dataset_pipelines)
    output_names = all_dataset_pipelines.all_outputs()
    input_names = all_dataset_pipelines.inputs()
    data_input_names = [
        name for name in input_names if "test" in name or "train" in name
    ]

    consolidated_model_pipelines = pipeline(
        pipe=all_dataset_pipelines,
        inputs={k: k for k in data_input_names},
        outputs=output_names,
        namespace="train_evaluation",
    )

    return consolidated_model_pipelines
