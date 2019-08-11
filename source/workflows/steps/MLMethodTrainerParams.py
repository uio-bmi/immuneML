from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.ml_methods.MLMethod import MLMethod
from source.workflows.steps.StepParams import StepParams


class MLMethodTrainerParams(StepParams):

    def __init__(self, method: MLMethod, dataset: RepertoireDataset, result_path: str, labels: list, model_selection_cv: bool,
                 model_selection_n_folds: int, cores_for_training: int):
        self.method = method
        self.result_path = result_path
        self.dataset = dataset
        self.labels = labels
        self.model_selection_cv = model_selection_cv
        self.model_selection_n_folds = model_selection_n_folds
        self.cores_for_training = cores_for_training
