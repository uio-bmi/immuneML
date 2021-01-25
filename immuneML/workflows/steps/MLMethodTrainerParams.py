from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.workflows.steps.StepParams import StepParams


class MLMethodTrainerParams(StepParams):

    def __init__(self, method: MLMethod, dataset: Dataset, result_path: Path, label: str, model_selection_cv: bool,
                 model_selection_n_folds: int, cores_for_training: int, train_predictions_path: Path, ml_details_path: Path,
                 optimization_metric: str):
        self.method = method
        self.result_path = result_path
        self.dataset = dataset
        self.label = label
        self.model_selection_cv = model_selection_cv
        self.model_selection_n_folds = model_selection_n_folds
        self.cores_for_training = cores_for_training
        self.train_predictions_path = train_predictions_path
        self.ml_details_path = ml_details_path
        self.optimization_metric = optimization_metric
