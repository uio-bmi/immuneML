import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from immuneML import Constants


def specificity_score(true_y, predicted_y, sample_weight=None, labels=None):
    tn, fp, fn, tp = confusion_matrix(true_y, predicted_y, labels=labels, sample_weight=sample_weight).ravel()
    return tn / (tn + fp) if tn + fp != 0 else Constants.NOT_COMPUTED


def brier_score_loss(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.brier_score_loss(true_y, predicted_y, sample_weight=sample_weight)


def precision_score_micro(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.precision_score(true_y, predicted_y, average="micro", sample_weight=sample_weight, labels=labels)


def precision_score_macro(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.precision_score(true_y, predicted_y, average="macro", sample_weight=sample_weight, labels=labels)


def precision_score_weighted(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.precision_score(true_y, predicted_y, average="weighted", sample_weight=sample_weight, labels=labels)


def recall_score_micro(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.recall_score(true_y, predicted_y, average="micro", sample_weight=sample_weight, labels=labels)


def recall_score_macro(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.recall_score(true_y, predicted_y, average="macro", sample_weight=sample_weight, labels=labels)


def recall_score_weighted(true_y, predicted_y, sample_weight=None, labels=None):
    return metrics.recall_score(true_y, predicted_y, average="weighted", sample_weight=sample_weight, labels=labels)


def f1_score_weighted(true_y, predicted_y, sample_weight=None):
    return metrics.f1_score(true_y, predicted_y, average="weighted", sample_weight=sample_weight)


def f1_score_micro(true_y, predicted_y, sample_weight=None):
    return metrics.f1_score(true_y, predicted_y, average="micro", sample_weight=sample_weight)


def f1_score_macro(true_y, predicted_y, sample_weight=None):
    return metrics.f1_score(true_y, predicted_y, average="macro", sample_weight=sample_weight)


def roc_auc_score_ovo(true_y, predicted_y, sample_weight=None, labels=None):
    return roc_auc_score(true_y, predicted_y, sample_weight=sample_weight, labels=labels, multiclass='ovo')


def roc_auc_score_ovr(true_y, predicted_y, sample_weight=None, labels=None):
    return roc_auc_score(true_y, predicted_y, sample_weight=sample_weight, labels=labels, multiclass='ovr')


def roc_auc_score(true_y, predicted_y, sample_weight=None, labels=None, multiclass: str = 'raise'):
    predictions = np.array(predicted_y) if not isinstance(predicted_y, np.ndarray) else predicted_y
    true_values = np.array(true_y) if not isinstance(true_y, np.ndarray) else true_y
    if predictions.shape == true_values.shape:
        return metrics.roc_auc_score(true_values, predictions, sample_weight=sample_weight, labels=labels,
                                     multi_class=multiclass)
    elif len(predictions.shape) == 2 and predictions.shape[1] == 2:
        return metrics.roc_auc_score(true_values, predictions[:, 1], sample_weight=sample_weight, labels=labels,
                                     multi_class=multiclass)
    else:
        return -1
