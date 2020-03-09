from source.data_model.dataset.Dataset import Dataset
from source.environment.MetricType import MetricType
from source.ml_methods.MLMethod import MLMethod
from source.workflows.steps.StepParams import StepParams


class MLMethodAssessmentParams(StepParams):

    def __init__(self, method: MLMethod, dataset: Dataset, metrics: set, optimization_metric: MetricType, label: str,
                 path: str, split_index: int, predictions_path: str, ml_score_path: str):
        self.method = method
        self.dataset = dataset
        self.metrics = metrics
        self.optimization_metric = optimization_metric
        self.path = path
        self.label = label
        self.split_index = split_index
        self.predictions_path = predictions_path
        self.ml_score_path = ml_score_path
