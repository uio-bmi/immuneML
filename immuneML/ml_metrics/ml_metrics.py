import numpy as np
from sklearn import metrics


def f1_score_weighted(true_y, predicted_y, sample_weight=None):
    return metrics.f1_score(true_y, predicted_y, average="weighted", sample_weight=sample_weight)


def f1_score_micro(true_y, predicted_y, sample_weight=None):
    return metrics.f1_score(true_y, predicted_y, average="micro", sample_weight=sample_weight)


def f1_score_macro(true_y, predicted_y, sample_weight=None):
    return metrics.f1_score(true_y, predicted_y, average="macro", sample_weight=sample_weight)


def roc_auc_score(true_y, predicted_y, sample_weight=None, labels=None):
    predictions = np.array(predicted_y) if not isinstance(predicted_y, np.ndarray) else predicted_y
    true_values = np.array(true_y) if not isinstance(true_y, np.ndarray) else true_y
    if predictions.shape == true_values.shape:
        return metrics.roc_auc_score(true_values, predictions, sample_weight=sample_weight, labels=labels)
    elif len(predictions.shape) == 2 and predictions.shape[1] == 2:
        return metrics.roc_auc_score(true_values, predictions[:, 1], sample_weight=sample_weight, labels=labels)
    else:
        return -1
