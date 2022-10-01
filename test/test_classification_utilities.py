from sensor_util.classification_utilities import (
    calculate_sensitivity_and_specificity,
    print_classification_report_binary,
    oversample_minority_class,
)

import pandas as pd
import numpy as np


def test_print_classification_report_binary():
    y_predicted = pd.Series([0, 1, 1, 0, 0, 0])
    y_test = pd.Series([0, 1, 0, 0, 1, 0])
    predicted_score = np.array(
        [[0.7, 0.3], [0.1, 0.9], [0.4, 0.6], [0.65, 0.35], [0.6, 0.4], [0.8, 0.2]]
    )
    print_classification_report_binary(y_test, y_predicted, predicted_score)


def test_calculate_sensitivity_and_specificity():
    y_predicted = pd.Series([0, 1, 1, 0, 0, 0])
    y_test = pd.Series([0, 1, 0, 0, 1, 0])

    s = calculate_sensitivity_and_specificity(y_test, y_predicted)
    sensitivity = s["sensitivity"]
    specificity = s["specificity"]
    assert sensitivity == 1 / 2
    assert specificity == 3 / 4

    y_predicted = pd.Series([1, 1, 2, 1, 0, 0])
    y_test = pd.Series([2, 1, 2, 1, 1, 0])

    s = calculate_sensitivity_and_specificity(y_test, y_predicted, positive_condition=2)
    sensitivity = s["sensitivity"]
    specificity = s["specificity"]
    assert sensitivity == 1 / 2
    assert specificity == 4 / 4

    y_predicted = pd.Series([0, 0, 1])
    y_test = pd.Series([0, 0, 0])
    s = calculate_sensitivity_and_specificity(y_test, y_predicted, positive_condition=1)
    sensitivity = s["sensitivity"]
    specificity = s["specificity"]
    assert np.isnan(sensitivity)
    assert specificity == 2 / 3

    y_predicted = pd.Series([0, 1, 1])
    y_test = pd.Series([1, 1, 1])
    s = calculate_sensitivity_and_specificity(y_test, y_predicted, positive_condition=1)
    sensitivity = s["sensitivity"]
    specificity = s["specificity"]
    assert sensitivity == 2 / 3
    assert np.isnan(specificity)


def test_oversample_minority_class():
    feat1 = pd.Series([0.2, 0.3, 0.4, 0.5])
    feat2 = pd.Series([10, -20, 30, 50])
    X_train = pd.DataFrame({"feature_1": feat1, "feature_2": feat2})
    y_train = pd.DataFrame({"classification_feature": pd.Series([1, 1, 1, 0])})
    num_additional_samples = 3
    minority_class = 0
    x_train_augmented, y_train_augmented = oversample_minority_class(
        X_train, y_train, num_additional_samples, minority_class
    )

    assert x_train_augmented.shape[0] == 7
    assert x_train_augmented.shape[1] == 2
    assert y_train_augmented.shape[0] == 7
    assert y_train_augmented.value_counts()[0] == 4
    assert y_train_augmented.value_counts()[1] == 3
