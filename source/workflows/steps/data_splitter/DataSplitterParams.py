from dataclasses import dataclass

from source.data_model.dataset.Dataset import Dataset
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.workflows.steps.StepParams import StepParams


@dataclass
class DataSplitterParams(StepParams):

    dataset: Dataset
    split_strategy: SplitType
    split_count: int
    training_percentage: float = -1
    paths: list = None
    split_config: SplitConfig = None
