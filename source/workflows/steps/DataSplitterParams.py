from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.hyperparameter_optimization.SplitType import SplitType
from source.workflows.steps.StepParams import StepParams


class DataSplitterParams(StepParams):

    def __init__(self, dataset: RepertoireDataset, split_strategy: SplitType, split_count: int, training_percentage: float = -1,
                 label_to_balance: str = None):
        self.dataset = dataset
        self.split_strategy = split_strategy
        self.split_count = split_count
        self.training_percentage = training_percentage
        self.label_to_balance = label_to_balance
