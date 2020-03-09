import inspect
import os
import warnings

import pandas as pd
from sklearn import metrics

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.MetricType import MetricType
from source.ml_methods.MLMethod import MLMethod
from source.ml_metrics import ml_metrics
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from source.workflows.steps.Step import Step


class MLMethodAssessment(Step):

    fieldnames = ["run", "optimal_method_params", "method", "encoding_params", "encoding", "evaluated_on"]

    @staticmethod
    def run(input_params: MLMethodAssessmentParams = None):
        X = input_params.dataset.encoded_data.examples
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
                                            label=input_params.label, split_index=input_params.split_index,
                                            predicted_y=predicted_y, true_y=true_y, method=input_params.method,
                                            dataset=input_params.dataset, ml_score_path=input_params.ml_score_path)

        summary_metric = MLMethodAssessment._get_optimization_metric(results, input_params.label, input_params.optimization_metric)

        return summary_metric

    @staticmethod
    def _get_optimization_metric(df: pd.DataFrame, label: str, metric: MetricType) -> float:
        return df["{}_{}".format(label, metric.name.lower())].iloc[0]

    @staticmethod
    def _score(metrics_list: list, optimization_metric: MetricType, label: str, predicted_y, true_y, ml_score_path: str, split_index: int,
               method: MLMethod, dataset: RepertoireDataset):
        results = {}

        metrics_with_optim_metric = set(metrics_list)
        metrics_with_optim_metric.add(optimization_metric)

        for metric in list(metrics_with_optim_metric):
            score = MLMethodAssessment._score_for_metric(metric, predicted_y[label], true_y[label],
                                                         method.get_classes_for_label(label))
            results["{}_{}".format(label, metric.name.lower())] = score

        results["split_index"] = split_index
        results["{}_method_params".format(label)] = {**method.get_params(label),
                                                     "feature_names": dataset.encoded_data.feature_names
                                                     if dataset is not None else None}

        df = pd.DataFrame([results])

        if os.path.isfile(ml_score_path) and os.path.getsize(ml_score_path) > 0:
            df.to_csv(ml_score_path, mode='a', header=False, index=False)
        else:
            df.to_csv(ml_score_path, index=False)

        return df

    @staticmethod
    def _score_for_metric(metric: MetricType, predicted_y, true_y, labels):
        if hasattr(metrics, metric.value) and callable(getattr(metrics, metric.value)):
            fn = getattr(metrics, metric.value)
        else:
            fn = getattr(ml_metrics, metric.value)

        try:
            if "labels" in inspect.signature(fn).parameters.keys():
                score = fn(true_y, predicted_y, labels=labels)
            else:
                score = fn(true_y, predicted_y)
        except ValueError:
            warnings.warn(f"MLMethodAssessment: score for metric {metric.name} could not be calculated."
                          f"\nPredicted values: {predicted_y}\nTrue values: {true_y}", RuntimeWarning)
            score = None

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
