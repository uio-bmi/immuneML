import json
from itertools import product
from multiprocessing.pool import Pool
from sklearn import metrics

from source.ml_metrics import ml_metrics
from source.environment.MetricType import MetricType
from source.environment.ParallelismManager import ParallelismManager
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.Step import Step


def assess_per_method(method: MLMethod, input_params: dict):
    return MLMethodAssessment.assess_per_method(method, input_params)


class MLMethodAssessment(Step):

    @staticmethod
    def run(input_params: dict = None):
        MLMethodAssessment.check_prerequisites(input_params)
        return MLMethodAssessment.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, "MLMethodAssessment: input_params have to be set."
        assert "methods" in input_params, "MLMethodAssessment: the methods parameter has to be set to a list of trained ML method instances."
        assert "dataset" in input_params, "MLMethodAssessment: the dataset parameter has to be set to contain an instance of a Dataset object to be used for testing."
        assert "metrics" in input_params, "MLMethodAssessment: the metrics parameter has to be set to a list of metrics to be evaluated for the methods."

    @staticmethod
    def perform_step(input_params: dict = None):
        assessment_result = {}

        n_jobs = ParallelismManager.assign_cores_to_job("ml_method_assessment")
        with Pool(n_jobs) as pool:
            results = pool.starmap(assess_per_method, product(input_params["methods"], [input_params]))

        outputs = [result for result in results]

        for index, method in enumerate(input_params["methods"]):
            assessment_result[method.__class__.__name__] = outputs[index]

        return assessment_result

    @staticmethod
    def assess_per_method(method: MLMethod, input_params: dict):

        labels = input_params["labels"]
        X = input_params["dataset"].encoded_data["repertoires"]
        predicted_y = method.predict(X, labels)
        true_y = input_params["dataset"].encoded_data["labels"]

        MLMethodAssessment._store_predictions(method.__class__.__name__,
                                               true_y,
                                               predicted_y,
                                               labels,
                                               input_params["predictions_path"])

        results = {label: {} for label in labels}

        for metric in input_params["metrics"]:
            for index, label in enumerate(labels):
                score = MLMethodAssessment._score(metric, predicted_y[label], true_y[index])
                results[label][metric.name] = score

        return results

    @staticmethod
    def _score(metric: MetricType, predicted_y, true_y):
        if hasattr(metrics, metric.value) and callable(getattr(metrics, metric.value)):
            score = getattr(metrics, metric.value)(true_y, predicted_y)
        else:
            score = getattr(ml_metrics, metric.value)(true_y, predicted_y)

        return score


    @staticmethod
    def _store_predictions(method_name, true_y, predicted_y, labels, predictions_path):

        if predictions_path is not None:
            PathBuilder.build(predictions_path)

            obj = []

            for index, label in enumerate(labels):
                obj.append({
                    "label": label,
                    "predicted_y": predicted_y[label].tolist(),
                    "true_y": true_y[index].tolist()
                })

            with open(predictions_path + method_name + ".json", "w") as file:
                json.dump(obj, file, indent=2)
