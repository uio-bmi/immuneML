from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.Metric import Metric
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.workflows.steps.StepParams import StepParams


class MLMethodAssessmentParams(StepParams):

    def __init__(self, method: MLMethod, dataset: Dataset, metrics: set, optimization_metric: Metric, label: str,
                 path: Path, split_index: int, predictions_path: Path, ml_score_path: Path):
        self.method = method
        self.dataset = dataset
        self.metrics = metrics
        self.optimization_metric = optimization_metric
        self.path = path
        self.label = label
        self.split_index = split_index
        self.predictions_path = predictions_path
        self.ml_score_path = ml_score_path
