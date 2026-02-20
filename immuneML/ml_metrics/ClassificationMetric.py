from enum import Enum


class ClassificationMetric(Enum):

    ACCURACY = "accuracy_score"
    BALANCED_ACCURACY = "balanced_accuracy_score"
    CONFUSION_MATRIX = "confusion_matrix"
    F1_MICRO = "f1_score_micro"
    F1_MACRO = "f1_score_macro"
    F1_WEIGHTED = "f1_score_weighted"
    PRECISION = "precision_score"
    PRECISION_MICRO = "precision_score_micro"
    PRECISION_MACRO = "precision_score_macro"
    PRECISION_WEIGHTED = "precision_score_weighted"
    RECALL_MICRO = "recall_score_micro"
    RECALL_MACRO = "recall_score_macro"
    RECALL_WEIGHTED = "recall_score_weighted"
    AVERAGE_PRECISION = "average_precision_score"
    BRIER_SCORE = "brier_score_loss"
    RECALL = "recall_score"
    AUC = "roc_auc_score"
    AUC_OVO = "roc_auc_score_ovo"
    AUC_OVR = "roc_auc_score_ovr"
    LOG_LOSS = "log_loss"
    SENSITIVITY = "recall_score"  # Sensitivity is equivalent to recall
    SPECIFICITY = "specificity_score"  # Specificity needs to be implemented separately

    @staticmethod
    def get_binary_only_metrics():
        """Metrics that required binarized labels"""
        return {
            ClassificationMetric.AUC,
            ClassificationMetric.AVERAGE_PRECISION,
            ClassificationMetric.BRIER_SCORE,
            ClassificationMetric.AUC_OVR,
            ClassificationMetric.AUC_OVO
        }

    @staticmethod
    def get_metric(metric_name: str):
        try:
            return ClassificationMetric[metric_name.upper()]
        except KeyError as e:
            raise KeyError(f"'{metric_name}' is not a valid performance metric. Valid metrics are: {', '.join([m.name for m in ClassificationMetric])}").with_traceback(e.__traceback__)

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