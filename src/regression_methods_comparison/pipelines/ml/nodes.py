import importlib
import logging
import random
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_options: Dict[str, Any]
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Trains a regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    logger = logging.getLogger(__name__)

    model_module = model_options["module"]
    model_type = model_options["class"]
    model_arguments = model_options["kwargs"]

    regressor_class = getattr(importlib.import_module(model_module), model_type)

    if model_arguments:
        regressor_instance = regressor_class(**model_arguments)
    else:
        logger.info(
            f"No arguments provided for model: {model_type}. Continuing with default arguments."
        )
        regressor_instance = regressor_class()

    logger.info(f"Fitting model of type {type(regressor_instance)}")

    regressor_instance.fit(X_train, y_train)

    flat_model_params = {
        **{"model_type": model_type},
        **regressor_instance.get_params(),
    }
    return regressor_instance, flat_model_params


def evaluate(
    linreg: LinearRegression,
    knn: KNeighborsRegressor,
    random_forest: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    models = {"linear regression": linreg, "KNN": knn, "random forest": random_forest}

    rows = []
    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "model": model_name,
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "rmse": rmse,
            "r2": r2,
        }
        rows.append(metrics)

    metrics = pd.DataFrame(rows)
    return metrics


def evaluate_model(
    regressor: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    a2_score = random.randint(0, 100) * 0.1
    b2_score = random.randint(0, 100) * 0.1
    logger = logging.getLogger(__name__)
    logger.info(
        f"Model has a coefficient R^2 of {score:.3f} on test data using a "
        f"regressor of type '{type(regressor)}'"
    )
    return {"r2_score": score, "a2_score": a2_score, "b2_score": b2_score}
