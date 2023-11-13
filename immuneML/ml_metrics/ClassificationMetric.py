from enum import Enum


class ClassificationMetric(Enum):

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
    def get_metric(metric_name: str):
        try:
            return ClassificationMetric[metric_name.upper()]
        except KeyError:
            raise KeyError(f"'{metric_name}' is not a valid performance metric. Valid metrics are: {', '.join([m.name for m in ClassificationMetric])}")

    @staticmethod
    def get_search_criterion(metric):
        if metric in [ClassificationMetric.LOG_LOSS]:
            return min
        else:
            return max

    @staticmethod
    def get_sklearn_score_name(metric):
        if metric in [ClassificationMetric.LOG_LOSS]:
            return f"neg_{metric.name.lower()}"
        else:
            return metric.name.lower()

    @staticmethod
    def get_probability_based_metric_types():
        return [ClassificationMetric.LOG_LOSS, ClassificationMetric.AUC]