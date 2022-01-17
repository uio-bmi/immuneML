import os
import warnings
from pathlib import Path

import pandas as pd
from sklearn import metrics

from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.ml_metrics import ml_metrics
from immuneML.ml_metrics.Metric import Metric
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from immuneML.workflows.steps.Step import Step


class MLMethodAssessment(Step):
    fieldnames = ["run", "optimal_method_params", "method", "encoding_params", "encoding", "evaluated_on"]

    @staticmethod
    def run(input_params: MLMethodAssessmentParams = None):
        X = input_params.dataset.encoded_data
        predicted_y = input_params.method.predict(X, input_params.label)
        predicted_proba_y = input_params.method.predict_proba(X, input_params.label)
        true_y = input_params.dataset.encoded_data.labels

        example_ids = input_params.dataset.get_example_ids()

        MLMethodAssessment._store_predictions(method=input_params.method, true_y=true_y, predicted_y=predicted_y, predicted_proba_y=predicted_proba_y,
                                              label_name=input_params.label.name, predictions_path=input_params.predictions_path,
                                              example_ids=example_ids, split_index=input_params.split_index)

        scores = MLMethodAssessment._score(metrics_list=input_params.metrics, optimization_metric=input_params.optimization_metric,
                                           label_name=input_params.label.name, split_index=input_params.split_index, predicted_y=predicted_y,
                                           predicted_proba_y=predicted_proba_y, true_y=true_y, method=input_params.method,
                                           ml_score_path=input_params.ml_score_path)

        return scores

    @staticmethod
    def _score(metrics_list: set, optimization_metric: Metric, label_name: str, predicted_y, predicted_proba_y, true_y, ml_score_path: Path,
               split_index: int, method: MLMethod):
        results = {}
        scores = {}

        metrics_with_optim_metric = set(metrics_list)
        metrics_with_optim_metric.add(optimization_metric)

        metrics_with_optim_metric = sorted(list(metrics_with_optim_metric), key=lambda metric: metric.name)

        for metric in metrics_with_optim_metric:
            predicted_proba_y_label = predicted_proba_y[label_name] if predicted_proba_y is not None else None
            score = MLMethodAssessment._score_for_metric(metric=metric, predicted_y=predicted_y[label_name], true_y=true_y[label_name],
                                                         classes=method.get_classes(),
                                                         predicted_proba_y=predicted_proba_y_label)
            results[f"{label_name}_{metric.name.lower()}"] = score
            scores[metric.name.lower()] = score

        results["split_index"] = split_index

        df = pd.DataFrame([results])

        if ml_score_path.is_file() and os.path.getsize(ml_score_path) > 0:
            df.to_csv(ml_score_path, mode='a', header=False, index=False)
        else:
            df.to_csv(ml_score_path, index=False)

        return scores

    @staticmethod
    def _score_for_metric(metric: Metric, predicted_y, predicted_proba_y, true_y, classes):
        if hasattr(ml_metrics, metric.value):
            fn = getattr(ml_metrics, metric.value)
        else:
            fn = getattr(metrics, metric.value)

        true_y, predicted_y = Util.binarize_label_classes(true_y=true_y, predicted_y=predicted_y, classes=classes)

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
            score = "not computed"

        return score

    @staticmethod
    def _store_predictions(method: MLMethod, true_y, predicted_y, predicted_proba_y, label_name: str, predictions_path, summary_path=None,
                           example_ids: list = None, split_index: int = None):

        df = pd.DataFrame()
        df["example_id"] = example_ids
        df["split_index"] = [split_index for i in range(len(example_ids))]

        df[f"{label_name}_true_class"] = true_y[label_name]
        df[f"{label_name}_predicted_class"] = predicted_y[label_name]

        classes = method.get_classes()
        for cls_index, cls in enumerate(classes):
            tmp = predicted_proba_y[label_name][:, cls_index] if predicted_proba_y is not None and predicted_proba_y[label_name] is not None else None
            df[f"{label_name}_{cls}_proba"] = tmp

        if predictions_path is not None:
            df.to_csv(predictions_path, index=False)

        if summary_path is not None:
            PathBuilder.build(os.path.dirname(os.path.abspath(summary_path)))
            if os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0:
                df.to_csv(summary_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_path, index=False)
