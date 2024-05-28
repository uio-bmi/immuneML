import os
from pathlib import Path

import numpy as np
import pandas as pd
from immuneML.ml_metrics.MetricUtil import MetricUtil

from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from immuneML.workflows.steps.Step import Step


class MLMethodAssessment(Step):
    fieldnames = ["run", "optimal_method_params", "method", "encoding_params", "encoding", "evaluated_on"]

    @staticmethod
    def run(input_params: MLMethodAssessmentParams = None):
        X = input_params.dataset.encoded_data
        predicted_y = input_params.method.predict(X, input_params.label)
        predicted_proba_y_per_class = input_params.method.predict_proba(X, input_params.label)
        true_y = input_params.dataset.encoded_data.labels
        example_weights = input_params.dataset.get_example_weights()

        example_ids = input_params.dataset.get_example_ids()

        MLMethodAssessment._store_predictions(method=input_params.method, true_y=true_y, predicted_y=predicted_y,
                                              predicted_proba_y_per_class=predicted_proba_y_per_class,
                                              label=input_params.label, predictions_path=input_params.predictions_path,
                                              example_ids=example_ids, split_index=input_params.split_index)

        scores = MLMethodAssessment._score(metrics_list=input_params.metrics, optimization_metric=input_params.optimization_metric,
                                           label=input_params.label, split_index=input_params.split_index, predicted_y=predicted_y,
                                           predicted_proba_y_per_class=predicted_proba_y_per_class, true_y=true_y,
                                           example_weights=example_weights, method=input_params.method,
                                           ml_score_path=input_params.ml_score_path)

        return scores

    @staticmethod
    def _score(metrics_list: set, optimization_metric: ClassificationMetric, label: Label, predicted_y, predicted_proba_y_per_class, true_y, example_weights, ml_score_path: Path,
               split_index: int, method: MLMethod):
        results = {}
        scores = {}

        predicted_proba_class = predicted_proba_y_per_class[label.name] if predicted_proba_y_per_class is not None else None
        predicted_proba_y = np.vstack([predicted_proba_class[cls] for cls in label.values]).T if predicted_proba_class is not None else None

        metrics_with_optim_metric = set(metrics_list)
        metrics_with_optim_metric.add(optimization_metric)

        metrics_with_optim_metric = sorted(list(metrics_with_optim_metric), key=lambda metric: metric.name)

        for metric in metrics_with_optim_metric:
            score = MetricUtil.score_for_metric(metric=metric,
                                                predicted_y=predicted_y[label.name],
                                                true_y=true_y[label.name],
                                                example_weights=example_weights,
                                                classes=label.values,
                                                predicted_proba_y=predicted_proba_y)
            results[f"{label.name}_{metric.name.lower()}"] = score
            scores[metric.name.lower()] = score

        results["split_index"] = split_index

        df = pd.DataFrame([results])

        if ml_score_path.is_file() and os.path.getsize(ml_score_path) > 0:
            df.to_csv(ml_score_path, mode='a', header=False, index=False)
        else:
            df.to_csv(ml_score_path, index=False)

        return scores

    @staticmethod
    def _store_predictions(method: MLMethod, true_y, predicted_y, predicted_proba_y_per_class, label: Label, predictions_path, summary_path=None,
                           example_ids: list = None, split_index: int = None):

        df = pd.DataFrame()
        df["example_id"] = example_ids
        df["split_index"] = [split_index for i in range(len(example_ids))]

        df[f"{label.name}_true_class"] = true_y[label.name]
        df[f"{label.name}_predicted_class"] = predicted_y[label.name]

        for cls in method.get_classes():
            tmp = predicted_proba_y_per_class[label.name][cls] if predicted_proba_y_per_class is not None and predicted_proba_y_per_class[label.name] is not None else None
            df[f'{label.name}_{cls}_proba'] = tmp

        if predictions_path is not None:
            df.to_csv(predictions_path, index=False)

        if summary_path is not None:
            PathBuilder.build(os.path.dirname(os.path.abspath(summary_path)))
            if os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0:
                df.to_csv(summary_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_path, index=False)
