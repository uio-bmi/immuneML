import inspect
import os

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
        labels = input_params.label_configuration.get_labels_by_name()
        X = input_params.dataset.encoded_data.examples
        predicted_y = input_params.method.predict(X, labels)
        predicted_proba_y = input_params.method.predict_proba(X, labels)
        true_y = input_params.dataset.encoded_data.labels
        repertoire_ids = [rep.identifier for rep in input_params.dataset.get_data()]

        MLMethodAssessment._store_predictions(input_params.method,
                                              true_y,
                                              predicted_y,
                                              predicted_proba_y,
                                              labels,
                                              input_params.predictions_path,
                                              input_params.all_predictions_path,
                                              repertoire_ids,
                                              input_params.run)

        results = MLMethodAssessment._score(metrics_list=input_params.metrics, labels=labels, run=input_params.run,
                                            label_config=input_params.label_configuration, predicted_y=predicted_y,
                                            true_y=true_y, ml_details_path=input_params.ml_details_path,
                                            method=input_params.method, dataset=input_params.dataset)

        summary_metric = MLMethodAssessment._get_summary_metric(results, labels)

        return summary_metric

    @staticmethod
    def _get_summary_metric(df: pd.DataFrame, labels: list) -> dict:
        return {label: df["{}_{}".format(label, MetricType.BALANCED_ACCURACY.name.lower())].iloc[0] for label in labels}

    @staticmethod
    def _score(metrics_list: list, labels: list, label_config, predicted_y, true_y, ml_details_path: str, run: int,
               method: MLMethod, dataset: RepertoireDataset):
        results = {}

        for metric in metrics_list:
            for index, label in enumerate(labels):
                score = MLMethodAssessment._score_for_metric(metric, predicted_y[label], true_y[label],
                                                             label_config.get_label_values(label))
                results["{}_{}".format(label, metric.name.lower())] = score

        results["run"] = run
        for label in labels:
            results["{}_method_params".format(label)] = {**method.get_params(label),
                                                         "feature_names": dataset.encoded_data.feature_names
                                                         if dataset is not None else None}

        df = pd.DataFrame([results])

        if os.path.isfile(ml_details_path) and os.path.getsize(ml_details_path) > 0:
            df.to_csv(ml_details_path, mode='a', header=False, index=False)
        else:
            df.to_csv(ml_details_path, index=False)

        return df

    @staticmethod
    def _score_for_metric(metric: MetricType, predicted_y, true_y, labels):
        if hasattr(metrics, metric.value) and callable(getattr(metrics, metric.value)):
            fn = getattr(metrics, metric.value)
        else:
            fn = getattr(ml_metrics, metric.value)

        if "labels" in inspect.signature(fn).parameters.keys():
            score = fn(true_y, predicted_y, labels=labels)
        else:
            score = fn(true_y, predicted_y)

        return score

    @staticmethod
    def _store_predictions(method, true_y, predicted_y, predicted_proba_y, labels, predictions_path, summary_path=None,
                           repertoire_ids: list = None, run: str = None):

        df = pd.DataFrame()
        df["example_id"] = repertoire_ids
        df["run"] = [run for i in range(len(repertoire_ids))]
        for index, label in enumerate(labels):
            df["{}_true_class".format(label)] = true_y[label]
            df["{}_predicted_class".format(label)] = predicted_y[label]

            classes = method.get_classes_for_label(label)
            for cls_index, cls in enumerate(classes):
                tmp = predicted_proba_y[label][:, cls_index] if predicted_proba_y is not None else None
                df["{}_{}_proba".format(label, cls)] = tmp

        if predictions_path is not None:
            df.to_csv(predictions_path, index=False)

        if summary_path is not None:
            PathBuilder.build(os.path.dirname(os.path.abspath(summary_path)))
            if os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0:
                df.to_csv(summary_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_path, index=False)
