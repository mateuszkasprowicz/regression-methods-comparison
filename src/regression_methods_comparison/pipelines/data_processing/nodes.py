from typing import Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer


def split_data(data: pd.DataFrame, parameters: dict[str, Any]) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters_data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data[parameters["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def impute_missing_values(X_train, X_test, y_train, y_test, parameters: dict[str, Any]) -> Tuple:
    if "imputer" not in parameters:
        return X_train, X_test, y_train, y_test

    imputers_params: dict[str, Any] = parameters["imputer"]
    imputers = []

    if "mode_imputer" in imputers_params:
        mode_columns = imputers_params["mode_imputer"]["columns"]
        imputers.append((SimpleImputer(strategy="most_frequent"), mode_columns))

    if "mean_imputer" in imputers_params:
        mean_columns = imputers_params["mean_imputer"]["columns"]
        imputers.append((SimpleImputer(strategy="mean"), mean_columns))

    if "median_imputer" in imputers_params:
        median_columns = imputers_params["median_imputer"]["columns"]
        imputers.append(( SimpleImputer(strategy="median"), median_columns))

    imputing_preprocessor = make_column_transformer(
        *imputers,
        remainder="passthrough",
        n_jobs=-1,
        verbose_feature_names_out=False,
    )

    X_train = imputing_preprocessor.fit_transform(X_train, y_train)
    X_test = imputing_preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


def encode_categories(X_train, X_test, y_train, y_test, parameters: dict[str, Any]) -> Tuple:
    if "encoder" not in parameters:
        return X_train, X_test, y_train, y_test

    encoders_params: dict[str, Any] = parameters["encoder"]
    random_state: int = parameters.get("random_state") or 1234
    encoders = []

    if "one_hot_encoder" in encoders_params:
        one_hot_columns = encoders_params["one_hot_encoder"]["columns"]
        encoders.append((OneHotEncoder(sparse_output=False), one_hot_columns))

    if "target_encoder" in encoders_params:
        target_columns = encoders_params["target_encoder"]["columns"]
        encoders.append((TargetEncoder(target_type="continuous", random_state=random_state), target_columns))

    if "drop" in encoders_params:
        drop_columns = encoders_params["target_encoder"]["columns"]
        encoders.append(("drop", drop_columns))

    encoding_preprocessor = make_column_transformer(
        *encoders,
        remainder="passthrough",
        n_jobs=-1,
        verbose_feature_names_out=False,
    )

    X_train = encoding_preprocessor.fit_transform(X_train, y_train)
    X_test = encoding_preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test
