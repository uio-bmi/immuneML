import copy

from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.semantic_model.MLResult import MLResult
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.ml_methods.MLMethod import MLMethod
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class MLProcess:
    """
    Class that implements the machine learning process:
        1. encodes the training dataset
        2. encodes the test dataset (using parameters learnt in step 1 if there are any such parameters)
        3. trains the ML method on encoded training dataset
        4. assesses the method's performance on encoded test dataset

    It performs the task for a given label configuration, and given list of metrics (used only in the assessment step).
    """

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, label: str,
                 encoder: DatasetEncoder, encoder_params: dict, method: MLMethod, ml_params: dict, metrics: set,
                 path: str, reports: list = None, min_example_count: int = 2, batch_size: int = 2, cores: int = -1,
                 train_predictions_path: str = None, val_predictions_path: str = None, ml_details_path: str = None,
                 ml_score_path: str = None, label_config: LabelConfiguration=None,):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.label = label
        self.label_config = label_config
        self.encoder = encoder
        self.encoder_params = encoder_params
        self.method = copy.deepcopy(method)
        self.ml_params = ml_params
        self.path = path
        self.cores_for_training = cores
        self.batch_size = batch_size
        assert all([isinstance(metric, MetricType) for metric in metrics]), \
            "MLProcess: metrics are not set to be an instance of MetricType."
        self.metrics = metrics
        self.metrics.add(MetricType.BALANCED_ACCURACY)
        self.min_example_count = min_example_count
        self.ml_details_path = ml_details_path
        self.ml_score_path = ml_score_path
        self.train_predictions_path = train_predictions_path
        self.val_predictions_path = val_predictions_path
        self.reports = reports if reports is not None else []

    def get_ML_result(self):
        return MLResult(self.path)

    def run(self, run_id: int):
        encoded_train = self._run_encoder(self.train_dataset, True)
        method = self._train_ml_method(encoded_train)
        if self.test_dataset.get_example_count() > 0:
            encoded_test = self._run_encoder(self.test_dataset, False)
            performance = self._assess_ml_method(method, encoded_test, run_id)
            self._run_reports(method, encoded_train, encoded_test, self.path + "reports/")
        else:
            performance = None
        return method, performance

    def _run_reports(self, method: MLMethod, train_dataset: Dataset, test_dataset: RepertoireDataset, path: str):
        for report in self.reports:
            tmp_report = copy.deepcopy(report)
            tmp_report.method = method
            tmp_report.train_dataset = train_dataset
            tmp_report.test_dataset = test_dataset
            tmp_report.result_path = path
            tmp_report.generate_report()

    def _assess_ml_method(self, method: MLMethod, encoded_test_dataset: Dataset, run: int):
        return MLMethodAssessment.run(MLMethodAssessmentParams(
            method=method,
            dataset=encoded_test_dataset,
            metrics=self.metrics,
            label=self.label,
            split_index=run,
            predictions_path=self.val_predictions_path,
            path=self.path,
            ml_score_path=self.ml_score_path
        ))

    def _run_encoder(self, train_dataset: Dataset, learn_model: bool):
        return DataEncoder.run(DataEncoderParams(
            dataset=train_dataset,
            encoder=self.encoder,
            encoder_params=EncoderParams(
                model=self.encoder_params,
                result_path=self.path,
                batch_size=self.batch_size,
                label_configuration=self.label_config,
                learn_model=learn_model,
                filename="train_dataset.pkl" if learn_model else "test_dataset.pkl"
            )
        ))

    def _train_ml_method(self, encoded_train_dataset: Dataset) -> MLMethod:
        return MLMethodTrainer.run(MLMethodTrainerParams(
            method=self.method,
            result_path=self.path + "/ml_method/",
            dataset=encoded_train_dataset,
            label=self.label,
            model_selection_cv=self.ml_params["model_selection_cv"],
            model_selection_n_folds=self.ml_params["model_selection_n_folds"],
            cores_for_training=self.cores_for_training,
            train_predictions_path=self.train_predictions_path,
            ml_details_path=self.ml_details_path
        ))
