from source.data_model.dataset.Dataset import Dataset
from source.environment.LabelConfiguration import LabelConfiguration
from source.workflows.steps.StepParams import StepParams


class MLMethodAssessmentParams(StepParams):

    def __init__(self, method: dict, dataset: Dataset, metrics: set, label_configuration: LabelConfiguration,
                 path: str, run: int, ml_details_path: str, predictions_path: str, all_predictions_path: str):
        self.method = method
        self.dataset = dataset
        self.metrics = metrics
        self.path = path
        self.label_configuration = label_configuration
        self.run = run
        self.ml_details_path = ml_details_path
        self.predictions_path = predictions_path
        self.all_predictions_path = all_predictions_path
