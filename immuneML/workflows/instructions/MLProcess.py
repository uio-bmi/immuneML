import copy
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.MLMethodTrainer import MLMethodTrainer
from immuneML.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class MLProcess:
    """
    Class that implements the machine learning process:
        1. encodes the training dataset
        2. encodes the test dataset (using parameters learnt in step 1 if there are any such parameters)
        3. trains the ML method on encoded training dataset
        4. assesses the method's performance on encoded test dataset

    It performs the task for a given label configuration, and given list of metrics (used only in the assessment step).
    """

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, label: Label, metrics: set, optimization_metric: ClassificationMetric,
                 path: Path, ml_reports: List[MLReport] = None, encoding_reports: list = None, data_reports: list = None, number_of_processes: int = 2,
                 label_config: LabelConfiguration = None, report_context: dict = None, hp_setting: HPSetting = None, example_weighting: ExampleWeightingStrategy = None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.label = label
        self.label_config = label_config
        self.method = copy.deepcopy(hp_setting.ml_method)
        self.path = PathBuilder.build(path) if path is not None else None
        self.ml_settings_export_path = path / "ml_settings_config" if path is not None else None
        self.ml_score_path = path / "ml_score.csv" if path is not None else None
        self.train_predictions_path = path / "train_predictions.csv" if path is not None else None
        self.test_predictions_path = path / "test_predictions.csv" if path is not None else None
        self.report_path = PathBuilder.build(path / "reports") if path is not None else None
        self.number_of_processes = number_of_processes
        assert all([isinstance(metric, ClassificationMetric) for metric in metrics]), \
            "MLProcess: metrics are not set to be an instance of Metric."
        self.metrics = metrics
        self.optimization_metric = optimization_metric
        self.ml_reports = ml_reports if ml_reports is not None else []
        self.encoding_reports = encoding_reports if encoding_reports is not None else []
        self.data_reports = data_reports if data_reports is not None else []
        self.report_context = report_context
        self.hp_setting = copy.deepcopy(hp_setting)
        self.example_weighting = example_weighting

    def _set_paths(self):
        if self.path is None:
            raise RuntimeError("MLProcess: path is not set, stopping execution...")
        self.ml_settings_export_path = self.path / "ml_settings_config"
        self.ml_score_path = self.path / "ml_score.csv"
        self.train_predictions_path = self.path / "train_predictions.csv"
        self.test_predictions_path = self.path / "test_predictions.csv"
        self.report_path = PathBuilder.build(self.path / "reports")

    def run(self, split_index: int) -> HPItem:

        print_log(f"Evaluating hyperparameter setting: {self.hp_setting}...", include_datetime=True)

        PathBuilder.build(self.path)
        self._set_paths()

        processed_dataset = HPUtil.preprocess_dataset(self.train_dataset, self.hp_setting.preproc_sequence, self.path / "preprocessed_train_dataset",
                                                      self.report_context)

        weighted_dataset = HPUtil.weight_examples(dataset=processed_dataset, weighting_strategy=self.example_weighting, path=self.path / "weighted_train_datasets",
                                                  learn_model=True, number_of_processes=self.number_of_processes)

        encoded_train_dataset = HPUtil.encode_dataset(weighted_dataset, self.hp_setting, self.path / "encoded_datasets", learn_model=True,
                                                      context=self.report_context, number_of_processes=self.number_of_processes,
                                                      label_configuration=self.label_config)

        method = self._train_method(encoded_train_dataset)

        encoding_train_results = ReportUtil.run_encoding_reports(encoded_train_dataset, self.encoding_reports, self.report_path / "encoding_train", self.number_of_processes)

        hp_item = self._assess_on_test_dataset(encoded_train_dataset, encoding_train_results, method, split_index)

        print_log(f"Completed hyperparameter setting {self.hp_setting}.\n", include_datetime=True)

        return hp_item

    def _train_method(self, dataset) -> MLMethod:
        method = MLMethodTrainer.run(MLMethodTrainerParams(
            method=copy.deepcopy(self.hp_setting.ml_method),
            result_path=self.path / "ml_method",
            dataset=dataset,
            label=self.label,
            train_predictions_path=self.train_predictions_path,
            model_selection_cv=self.hp_setting.ml_params["model_selection_cv"],
            model_selection_n_folds=self.hp_setting.ml_params["model_selection_n_folds"],
            cores_for_training=self.number_of_processes,
            optimization_metric=self.optimization_metric.name.lower()
        ))
        return method

    def _assess_on_test_dataset(self, encoded_train_dataset, encoding_train_results, method, split_index) -> HPItem:
        if self.test_dataset is not None and self.test_dataset.get_example_count() > 0:
            processed_test_dataset = HPUtil.preprocess_dataset(self.test_dataset, self.hp_setting.preproc_sequence,
                                                               self.path / "preprocessed_test_dataset")

            weighted_test_dataset = HPUtil.weight_examples(dataset=processed_test_dataset, weighting_strategy=self.example_weighting,
                                                           path=self.path / "weighted_test_datasets", learn_model=False,
                                                           number_of_processes=self.number_of_processes)

            encoded_test_dataset = HPUtil.encode_dataset(weighted_test_dataset, self.hp_setting, self.path / "encoded_datasets",
                                                         learn_model=False, context=self.report_context, number_of_processes=self.number_of_processes,
                                                         label_configuration=self.label_config)

            performance = HPUtil.assess_performance(method, self.metrics, self.optimization_metric, encoded_test_dataset, split_index, self.path,
                                                    self.test_predictions_path, self.label, self.ml_score_path)

            encoding_test_results = ReportUtil.run_encoding_reports(encoded_test_dataset, self.encoding_reports, self.report_path / "encoding_test", self.number_of_processes)

            model_report_results = ReportUtil.run_ML_reports(encoded_train_dataset, encoded_test_dataset, method, self.ml_reports,
                                                             self.report_path / "ml_method", self.hp_setting, self.label, self.number_of_processes, self.report_context)

            hp_item = HPItem(method=method, hp_setting=self.hp_setting, train_predictions_path=self.train_predictions_path,
                             test_predictions_path=self.test_predictions_path, train_dataset=self.train_dataset,
                             test_dataset=self.test_dataset, split_index=split_index, model_report_results=model_report_results,
                             encoding_train_results=encoding_train_results, encoding_test_results=encoding_test_results, performance=performance,
                             encoder=self.hp_setting.encoder, ml_settings_export_path=self.ml_settings_export_path)
        else:
            hp_item = HPItem(method=method, hp_setting=self.hp_setting, train_predictions_path=self.train_predictions_path,
                             test_predictions_path=None, train_dataset=self.train_dataset,
                             split_index=split_index, encoding_train_results=encoding_train_results, encoder=self.hp_setting.encoder,
                             ml_settings_export_path=self.ml_settings_export_path)

        return hp_item
