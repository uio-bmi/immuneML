import copy
import datetime
from typing import List

from source.data_model.dataset.Dataset import Dataset
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.core.HPUtil import HPUtil
from source.hyperparameter_optimization.states.HPItem import HPItem
from source.reports.ReportUtil import ReportUtil
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class MLProcess:
    """
    Class that implements the machine learning process:
        1. encodes the training dataset
        2. encodes the test dataset (using parameters learnt in step 1 if there are any such parameters)
        3. trains the ML method on encoded training dataset
        4. assesses the method's performance on encoded test dataset

    It performs the task for a given label configuration, and given list of metrics (used only in the assessment step).
    """

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, label: str, metrics: set, optimization_metric: Metric,
                 path: str, ml_reports: List[MLReport] = None, encoding_reports: list = None, data_reports: list = None, number_of_processes: int = 2,
                 label_config: LabelConfiguration = None, report_context: dict = None, hp_setting: HPSetting = None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.label = label
        self.label_config = label_config
        self.method = copy.deepcopy(hp_setting.ml_method)
        self.path = PathBuilder.build(path)
        self.number_of_processes = number_of_processes
        assert all([isinstance(metric, Metric) for metric in metrics]), \
            "MLProcess: metrics are not set to be an instance of Metric."
        self.metrics = metrics
        self.metrics.add(Metric.BALANCED_ACCURACY)
        self.optimization_metric = optimization_metric
        self.ml_details_path = f"{path}ml_details.yaml"
        self.ml_score_path = f"{path}ml_score.csv"
        self.train_predictions_path = f"{path}train_predictions.csv"
        self.test_predictions_path = f"{path}test_predictions.csv"
        self.report_path = PathBuilder.build(f"{path}reports/")
        self.ml_reports = ml_reports if ml_reports is not None else []
        self.encoding_reports = encoding_reports if encoding_reports is not None else []
        self.data_reports = data_reports if data_reports is not None else []
        self.report_context = report_context
        self.hp_setting = hp_setting

    def run(self, split_index: int) -> HPItem:

        print(f"{datetime.datetime.now()}: Evaluating hyperparameter setting: {self.hp_setting}...")

        PathBuilder.build(self.path)

        processed_dataset = HPUtil.preprocess_dataset(self.train_dataset, self.hp_setting.preproc_sequence, f"{self.path}preprocessed_train_dataset/")

        encoded_train_dataset = HPUtil.encode_dataset(processed_dataset, self.hp_setting, f"{self.path}encoded_datasets/", learn_model=True,
                                                      context=self.report_context, batch_size=self.number_of_processes,
                                                      label_configuration=self.label_config)

        method = HPUtil.train_method(self.label, encoded_train_dataset, self.hp_setting, self.path, self.train_predictions_path, self.ml_details_path)

        encoding_train_results = ReportUtil.run_encoding_reports(encoded_train_dataset, self.encoding_reports, f"{self.report_path}encoding_train/")

        hp_item = self._assess_on_test_dataset(encoded_train_dataset, encoding_train_results, method, split_index)

        print(f"{datetime.datetime.now()}: Completed hyperparameter setting {self.hp_setting}.\n")

        return hp_item

    def _assess_on_test_dataset(self, encoded_train_dataset, encoding_train_results, method, split_index) -> HPItem:
        if self.test_dataset is not None and self.test_dataset.get_example_count() > 0:
            processed_test_dataset = HPUtil.preprocess_dataset(self.test_dataset, self.hp_setting.preproc_sequence,
                                                               f"{self.path}preprocessed_test_dataset/")

            encoded_test_dataset = HPUtil.encode_dataset(processed_test_dataset, self.hp_setting, f"{self.path}encoded_datasets/",
                                                         learn_model=False, context=self.report_context, batch_size=self.number_of_processes,
                                                         label_configuration=self.label_config)

            performance = HPUtil.assess_performance(method, self.metrics, self.optimization_metric, encoded_test_dataset, split_index, self.path,
                                                    self.test_predictions_path, self.label, self.ml_score_path)

            encoding_test_results = ReportUtil.run_encoding_reports(encoded_test_dataset, self.encoding_reports, f"{self.report_path}encoding_test/")

            model_report_results = ReportUtil.run_ML_reports(encoded_train_dataset, encoded_test_dataset, method, self.ml_reports,
                                                             f"{self.report_path}ml_method/", self.hp_setting, self.label, self.report_context)

            hp_item = HPItem(method=method, hp_setting=self.hp_setting, train_predictions_path=self.train_predictions_path,
                             test_predictions_path=self.test_predictions_path, ml_details_path=self.ml_details_path, train_dataset=self.train_dataset,
                             test_dataset=self.test_dataset, split_index=split_index, model_report_results=model_report_results,
                             encoding_train_results=encoding_train_results, encoding_test_results=encoding_test_results, performance=performance)
        else:
            hp_item = HPItem(method=method, hp_setting=self.hp_setting, train_predictions_path=self.train_predictions_path,
                             test_predictions_path=None, ml_details_path=self.ml_details_path, train_dataset=self.train_dataset,
                             split_index=split_index, encoding_train_results=encoding_train_results)

        return hp_item
