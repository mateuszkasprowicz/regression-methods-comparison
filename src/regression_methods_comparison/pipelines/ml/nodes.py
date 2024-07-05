import importlib
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

_LOGGER = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_options: Dict[str, Any]
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Trains a regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the traget variable.

    Returns:
        Trained model.
    """

    model_module = model_options["module"]
    model_type = model_options["class"]
    model_arguments = model_options["kwargs"]

    regressor_class = getattr(importlib.import_module(model_module), model_type)

    if model_arguments:
        regressor_instance = regressor_class(**model_arguments)
    else:
        _LOGGER.info(
            f"No arguments provided for model: {model_type}. Continuing with default arguments."
        )
        regressor_instance = regressor_class()

    _LOGGER.info(f"Fitting model of type {type(regressor_instance)}")

    regressor_instance.fit(X_train, y_train)

    flat_model_params = {
        **{"model_type": model_type},
        **regressor_instance.get_params(),
    }
    return regressor_instance, flat_model_params


def evaluate_model(
    regressor: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculates the metrics.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target variable.
    """
    y_pred = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    _LOGGER.info(
        f"Model has a coefficient R^2 of {r2:.3f} on test data using a "
        f"regressor of type '{type(regressor)}'"
    )

    metrics = {
        "mae": mae,
        "mse": mse,
        "mape": mape,
        "rmse": rmse,
        "r2": r2,
    }

    return metrics
