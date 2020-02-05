from enum import Enum


class MetricType(Enum):

    ACCURACY = "accuracy_score"
    BALANCED_ACCURACY = "balanced_accuracy_score"
    CONFUSION_MATRIX = "confusion_matrix"
    F1_MICRO = "f1_score_micro"
    F1_MACRO = "f1_score_macro"
    F1_WEIGHTED = "f1_score_weighted"
    PRECISION = "precision_score"
    RECALL = "recall_score"
    AUC = "roc_auc_score"
