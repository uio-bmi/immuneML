import warnings
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
    def score_for_metric(metric: ClassificationMetric, predicted_y, predicted_proba_y, true_y, classes, example_weights=None):
        '''
        Note: when providing label classes, make sure the 'positive class' is sorted last.
        This sorting should be done automatically when accessing Label.values
        '''

        fn = MetricUtil.get_metric_fn(metric)

        true_y, predicted_y = Util.binarize_label_classes(true_y=true_y, predicted_y=predicted_y, classes=classes)

        try:
            if metric in ClassificationMetric.get_probability_based_metric_types():
                predictions = predicted_proba_y
                if predicted_proba_y is None:
                    warnings.warn(
                        f"MLMethodAssessment: metric {metric} is specified, but the chosen ML method does not output "
                        f"class probabilities. Using predicted classes instead...")
                    predictions = predicted_y
            else:
                predictions = predicted_y

            if 'labels' in inspect.getfullargspec(fn).kwonlyargs or 'labels' in inspect.getfullargspec(fn).args:
                score = fn(true_y, predictions, sample_weight=example_weights, labels=classes)
            else:
                score = fn(true_y, predictions, sample_weight=example_weights)

        except ValueError as err:
            warnings.warn(f"MLMethodAssessment: score for metric {metric.name} could not be calculated."
                          f"\nPredicted values: {predicted_y}\nTrue values: {true_y}.\nMore details: {err}",
                          RuntimeWarning)
            score = Constants.NOT_COMPUTED

        return score
