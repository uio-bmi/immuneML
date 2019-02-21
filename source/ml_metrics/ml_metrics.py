from sklearn import metrics


def f1_score_weighted(true_y, predicted_y):
    return metrics.f1_score(true_y, predicted_y, average="weighted")


def f1_score_micro(true_y, predicted_y):
    return metrics.f1_score(true_y, predicted_y, average="micro")


def f1_score_macro(true_y, predicted_y):
    return metrics.f1_score(true_y, predicted_y, average="macro")

