from immuneML.ml_metrics import Metric
from immuneML.ml_metrics import ml_metrics
from sklearn import metrics as sklearn_metrics


class MetricUtil:

    @staticmethod
    def get_metric_fn(metric: Metric) -> str:
        if hasattr(ml_metrics, metric.value):
            fn = getattr(ml_metrics, metric.value)
        else:
            fn = getattr(sklearn_metrics, metric.value)

        return fn

