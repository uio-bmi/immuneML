from dataclasses import dataclass

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.workflows.steps.StepParams import StepParams


@dataclass
class DataSplitterParams(StepParams):

    dataset: Dataset
    split_strategy: SplitType
    split_count: int
    training_percentage: float = -1
    paths: list = None
    split_config: SplitConfig = None
