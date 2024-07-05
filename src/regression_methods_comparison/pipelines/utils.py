from typing import List


TEST_TRAIN_REFS = ["X_train", "X_test", "y_train", "y_test"]


def make_suffixes(strings: List[str], suffix: str):
    return [name + suffix for name in strings]
