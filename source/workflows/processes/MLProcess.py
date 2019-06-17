import warnings

import numpy as np
import pandas as pd
import yaml

from source.data_model.dataset.Dataset import Dataset
from source.dsl.AssessmentType import AssessmentType
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataSplitter import DataSplitter
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer


class MLProcess:

    def __init__(self, dataset: Dataset, path: str, label_configuration: LabelConfiguration, encoder: DatasetEncoder,
                 encoder_params: dict, method: MLMethod, assessment_type: AssessmentType, metrics: list,
                 model_selection_cv: bool, model_selection_n_folds: int = None, training_percentage: float = None,
                 split_count: int = None, min_example_count: int = 1):
        self._dataset = dataset
        self._split_count = split_count
        self._training_percentage = training_percentage
        self._path = "{}{}/".format(path, assessment_type.name.lower())
        self._label_configuration = label_configuration
        self._encoder = encoder
        self._encoder_params = encoder_params
        self._method = method
        self._assessment_type = assessment_type
        self._model_selection_cv = model_selection_cv
        self._n_folds = model_selection_n_folds
        assert all([isinstance(metric, MetricType) for metric in metrics]), \
            "MLProcess: metrics are not set to be an instance of MetricType."
        self._metrics = metrics
        self._details_path = self._path + "ml_details.csv"
        self._all_predictions_path = self._path + "predictions.csv"
        self._min_example_count = min_example_count

    def run(self):
        train_datasets, test_datasets = self._run_data_splitter()
        if all(self._is_ml_possible(ds) for ds in train_datasets):
            path = self._run(train_datasets, test_datasets)
        else:
            path = ""
            warnings.warn("MLProcess: There were not enough examples (repertoires) to run machine learning.")
        return path

    def _run(self, train_datasets: list, test_datasets: list) -> str:

        for index in range(len(train_datasets)):
            self._run_for_setting(train_datasets[index], test_datasets[index], index)
        path = self._summarize_runs()

        return path

    def _summary_for_metric(self, metric, label) -> dict:
        df = pd.read_csv(self._details_path)

        column = "{}_{}".format(label, metric.name.lower())

        summary = {
            "max": max(df[column]),
            "min": min(df[column]),
            "mean": float(df[column].mean()),
            "median": float(df[column].median())
        }

        return summary

    def _summarize_runs(self) -> str:

        summary = {}
        for label in self._label_configuration.get_labels_by_name():
            summary[label] = {}
            for metric in self._metrics:
                summary[label][metric.name.lower()] = self._summary_for_metric(metric, label)

        file_path = "{}summary.yml".format(self._path)
        with open(file_path, "w") as file:
            yaml.dump(summary, file)

        return file_path

    def _run_for_setting(self, train_dataset: Dataset, test_dataset: Dataset, run: int):

        path = self._path + "run_{}/".format(run+1)
        PathBuilder.build(path)
        encoded_train = self._run_encoder(train_dataset, True, path)
        encoded_test = self._run_encoder(test_dataset, False, path)
        method = self._train_ml_method(encoded_train, path)
        self._assess_ml_method(method, encoded_test, run, path)

    def _is_ml_possible(self, dataset: Dataset) -> bool:
        valid = True
        labels = self._label_configuration.get_labels_by_name()
        index = len(labels) - 1
        metadata = self._get_metadata(dataset, labels)
        while valid and index >= 0:
            unique, counts = np.unique(metadata[labels[index]], return_counts=True)
            valid = valid and len(unique) > 1 and all(count >= self._min_example_count for count in counts) \
                    and all(el in dataset.params[labels[index]] for el in unique)
            index -= 1

        if valid is not True:
            warnings.warn("For label {}: there are not enough different examples to run a ML algorithm."
                          .format(labels[index]))

        return valid

    def _get_metadata(self, dataset: Dataset, labels):
        if dataset.metadata_path:
            return dataset.get_metadata(labels)
        else:
            metadata = {label: [] for label in labels}
            for rep in dataset.get_data():
                for label in labels:
                    metadata[label].append(rep.metadata.custom_params[label])
            return metadata

    def _assess_ml_method(self, method: MLMethod, encoded_test_dataset: Dataset, run: int, path: str):
        MLMethodAssessment.run({
            "method": method,
            "dataset": encoded_test_dataset,
            "metrics": self._metrics,
            "labels": self._label_configuration.get_labels_by_name(),
            "predictions_path": path + "/prediction/",
            "label_configuration": self._label_configuration,
            "run": run,
            "ml_details_path": self._details_path,
            "all_predictions_path": self._all_predictions_path
        })

    def _run_encoder(self, train_dataset: Dataset, infer_model: bool, path: str):
        return DataEncoder.run({
            "dataset": train_dataset,
            "encoder": self._encoder,
            "encoder_params": EncoderParams(
                model=self._encoder_params,
                result_path=path,
                model_path=path,
                vectorizer_path=path,
                scaler_path=path,
                pipeline_path=path,
                label_configuration=self._label_configuration,
                filename="train_dataset.pkl" if infer_model else "test_dataset.pkl"
            )
        })

    def _train_ml_method(self, encoded_train_dataset: Dataset, path: str) -> MLMethod:
        return MLMethodTrainer.run({
            "method": self._method,
            "result_path": path + "/ml_method/",
            "dataset": encoded_train_dataset,
            "labels": self._label_configuration.get_labels_by_name(),
            "model_selection_cv": self._model_selection_cv,
            "model_selection_n_folds": self._n_folds
        })

    def _run_data_splitter(self) -> tuple:
        params = {
            "dataset": self._dataset,
            "assessment_type": self._assessment_type.name
        }
        if self._assessment_type != AssessmentType.loocv:
            params["split_count"] = self._split_count  # ignored for loocv
        if self._training_percentage is not None:
            params["training_percentage"] = self._training_percentage
        return DataSplitter.run(params)
