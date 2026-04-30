import logging
import inspect

from immuneML.ml_metrics import ml_metrics
from sklearn import metrics as sklearn_metrics
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_methods.util.Util import Util
from immuneML.environment.Constants import Constants


class MetricUtil:

    @staticmethod
    def get_metric_fn(metric: ClassificationMetric):
        if hasattr(ml_metrics, metric.value):
            fn = getattr(ml_metrics, metric.value)
        else:
            fn = getattr(sklearn_metrics, metric.value)

        return fn

    @staticmethod
    def _get_fn_params(fn):
        # inspect.getfullargspec does not follow __wrapped__ for sklearn-decorated functions
        # in sklearn >= 1.2; inspect.signature does, so use it as the authoritative source.
        return set(inspect.signature(fn).parameters.keys())

    @staticmethod
    def score_for_metric(metric: ClassificationMetric, predicted_y, predicted_proba_y, true_y, classes, pos_class=None):
        """
        Note: when providing label classes, make sure the 'positive class' is sorted last.
        This sorting should be done automatically when accessing Label.values
        """

        fn = MetricUtil.get_metric_fn(metric)

        if metric in ClassificationMetric.get_binary_only_metrics():
            processed_true_y, processed_predicted_y = Util.binarize_label_classes(true_y=true_y, predicted_y=predicted_y, classes=classes)
        else:
            processed_true_y, processed_predicted_y = true_y, predicted_y

        try:
            if metric in ClassificationMetric.get_probability_based_metric_types():
                predictions = predicted_proba_y
                if predicted_proba_y is None:
                    logging.warning(
                        f"MLMethodAssessment: metric {metric} is specified, but the chosen ML method does not output "
                        f"class probabilities. Using predicted classes instead...")
                    predictions = processed_predicted_y
            else:
                predictions = processed_predicted_y

            fn_params = MetricUtil._get_fn_params(fn)
            if 'labels' in fn_params:
                if 'pos_label' in fn_params:
                    score = fn(processed_true_y, predictions, labels=classes, pos_label=pos_class if pos_class is not None else classes[-1])
                else:
                    score = fn(processed_true_y, predictions, labels=classes)
            else:
                score = fn(processed_true_y, predictions)

        except ValueError as err:
            logging.warning(f"MLMethodAssessment: score for metric {metric.name} could not be calculated."
                          f"\nPredicted values: {predicted_y}\nTrue values: {true_y}.\nMore details: {err}")
            score = Constants.NOT_COMPUTED

        return score
