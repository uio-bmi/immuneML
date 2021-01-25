from enum import Enum


class Metric(Enum):

    ACCURACY = "accuracy_score"
    BALANCED_ACCURACY = "balanced_accuracy_score"
    CONFUSION_MATRIX = "confusion_matrix"
    F1_MICRO = "f1_score_micro"
    F1_MACRO = "f1_score_macro"
    F1_WEIGHTED = "f1_score_weighted"
    PRECISION = "precision_score"
    RECALL = "recall_score"
    AUC = "roc_auc_score"
    LOG_LOSS = "log_loss"

    @staticmethod
    def get_search_criterion(metric):
        if metric in [Metric.LOG_LOSS]:
            return min
        else:
            return max

    @staticmethod
    def get_probability_based_metric_types():
        return [Metric.LOG_LOSS, Metric.AUC]
