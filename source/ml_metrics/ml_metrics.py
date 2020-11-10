import numpy as np
from sklearn import metrics


def f1_score_weighted(true_y, predicted_y):
    return metrics.f1_score(true_y, predicted_y, average="weighted")


def f1_score_micro(true_y, predicted_y):
    return metrics.f1_score(true_y, predicted_y, average="micro")


def f1_score_macro(true_y, predicted_y):
    return metrics.f1_score(true_y, predicted_y, average="macro")


def roc_auc_score(true_y, predicted_y):
    predictions = np.array(predicted_y) if not isinstance(predicted_y, np.ndarray) else predicted_y
    return metrics.roc_auc_score(true_y, predictions[:, 1] if len(predictions.shape) == 2 else predictions)
