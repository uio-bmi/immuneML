from pathlib import Path

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.workflows.steps.StepParams import StepParams


class MLMethodTrainerParams(StepParams):

    def __init__(self, method: MLMethod, dataset: Dataset, result_path: Path, label: Label, model_selection_cv: bool,
                 model_selection_n_folds: int, cores_for_training: int, train_predictions_path: Path,
                 optimization_metric: str):
        self.method = method
        self.result_path = result_path
        self.dataset = dataset
        self.label = label
        self.model_selection_cv = model_selection_cv
        self.model_selection_n_folds = model_selection_n_folds
        self.cores_for_training = cores_for_training
        self.train_predictions_path = train_predictions_path
        self.optimization_metric = optimization_metric
