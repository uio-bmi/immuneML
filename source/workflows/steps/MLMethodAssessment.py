import os
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from source.environment.Metric import Metric
from source.ml_methods.MLMethod import MLMethod
from source.ml_metrics import ml_metrics
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from source.workflows.steps.Step import Step


class MLMethodAssessment(Step):

    fieldnames = ["run", "optimal_method_params", "method", "encoding_params", "encoding", "evaluated_on"]

    @staticmethod
    def run(input_params: MLMethodAssessmentParams = None):
        X = input_params.dataset.encoded_data
        predicted_y = input_params.method.predict(X, [input_params.label])
        predicted_proba_y = input_params.method.predict_proba(X, [input_params.label])
        true_y = input_params.dataset.encoded_data.labels

        example_ids = input_params.dataset.get_example_ids()

        MLMethodAssessment._store_predictions(method=input_params.method,
                                              true_y=true_y,
                                              predicted_y=predicted_y,
                                              predicted_proba_y=predicted_proba_y,
                                              label=input_params.label,
                                              predictions_path=input_params.predictions_path,
                                              example_ids=example_ids,
                                              split_index=input_params.split_index)

        results = MLMethodAssessment._score(metrics_list=input_params.metrics, optimization_metric=input_params.optimization_metric,
                                            label=input_params.label, split_index=input_params.split_index, predicted_y=predicted_y,
                                            predicted_proba_y=predicted_proba_y, true_y=true_y, method=input_params.method,
                                            ml_score_path=input_params.ml_score_path)

        summary_metric = MLMethodAssessment._get_optimization_metric(results, input_params.label, input_params.optimization_metric)

        return summary_metric

    @staticmethod
    def _get_optimization_metric(df: pd.DataFrame, label: str, metric: Metric) -> float:
        return df["{}_{}".format(label, metric.name.lower())].iloc[0]

    @staticmethod
    def _score(metrics_list: list, optimization_metric: Metric, label: str, predicted_y, predicted_proba_y, true_y, ml_score_path: str,
               split_index: int, method: MLMethod):
        results = {}

        metrics_with_optim_metric = set(metrics_list)
        metrics_with_optim_metric.add(optimization_metric)

        for metric in list(metrics_with_optim_metric):
            predicted_proba_y_label = predicted_proba_y[label] if predicted_proba_y is not None else None
            score = MLMethodAssessment._score_for_metric(metric=metric, predicted_y=predicted_y[label], true_y=true_y[label],
                                                         labels=method.get_classes_for_label(label),
                                                         predicted_proba_y=predicted_proba_y_label)
            results["{}_{}".format(label, metric.name.lower())] = score

        results["split_index"] = split_index

        df = pd.DataFrame([results])

        if os.path.isfile(ml_score_path) and os.path.getsize(ml_score_path) > 0:
            df.to_csv(ml_score_path, mode='a', header=False, index=False)
        else:
            df.to_csv(ml_score_path, index=False)

        return df

    @staticmethod
    def _score_for_metric(metric: Metric, predicted_y, predicted_proba_y, true_y, labels):
        if hasattr(ml_metrics, metric.value):
            fn = getattr(ml_metrics, metric.value)
        else:
            fn = getattr(metrics, metric.value)

        if hasattr(true_y, 'dtype') and true_y.dtype.type is np.str_ or isinstance(true_y, list) and any(isinstance(item, str) for item in true_y):
            true_y = label_binarize(true_y, classes=labels)
            predicted_y = label_binarize(predicted_y, classes=labels)

        try:
            if metric in Metric.get_probability_based_metric_types():
                predictions = predicted_proba_y
                if predicted_proba_y is None:
                    warnings.warn(f"MLMethodAssessment: metric {metric} is specified, but the chosen ML method does not output "
                                  f"class probabilities. Using predicted classes instead...")
                    predictions = predicted_y
            else:
                predictions = predicted_y

            score = fn(true_y, predictions)

        except ValueError as err:
            warnings.warn(f"MLMethodAssessment: score for metric {metric.name} could not be calculated."
                          f"\nPredicted values: {predicted_y}\nTrue values: {true_y}.\nMore details: {err}", RuntimeWarning)
            score = -1.

        return score

    @staticmethod
    def _store_predictions(method, true_y, predicted_y, predicted_proba_y, label, predictions_path, summary_path=None,
                           example_ids: list = None, split_index: int = None):

        df = pd.DataFrame()
        df["example_id"] = example_ids
        df["split_index"] = [split_index for i in range(len(example_ids))]

        df["{}_true_class".format(label)] = true_y[label]
        df["{}_predicted_class".format(label)] = predicted_y[label]

        classes = method.get_classes_for_label(label)
        for cls_index, cls in enumerate(classes):
            tmp = predicted_proba_y[label][:, cls_index] if predicted_proba_y is not None and predicted_proba_y[label] is not None else None
            df["{}_{}_proba".format(label, cls)] = tmp

        if predictions_path is not None:
            df.to_csv(predictions_path, index=False)

        if summary_path is not None:
            PathBuilder.build(os.path.dirname(os.path.abspath(summary_path)))
            if os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0:
                df.to_csv(summary_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_path, index=False)
