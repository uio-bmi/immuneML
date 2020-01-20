from source.data_model.dataset.Dataset import Dataset
from source.ml_methods.MLMethod import MLMethod
from source.workflows.steps.StepParams import StepParams


class MLMethodAssessmentParams(StepParams):

    def __init__(self, method: MLMethod, dataset: Dataset, metrics: set, label: str,
                 path: str, split_index: int, predictions_path: str, ml_score_path: str):
        self.method = method
        self.dataset = dataset
        self.metrics = metrics
        self.path = path
        self.label = label
        self.split_index = split_index
        self.predictions_path = predictions_path
        self.ml_score_path = ml_score_path
