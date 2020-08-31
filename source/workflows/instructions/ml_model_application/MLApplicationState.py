from dataclasses import dataclass

from source.data_model.dataset.Dataset import Dataset
from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting


@dataclass
class MLApplicationState:

    dataset: Dataset
    hp_setting: HPSetting
    label_config: LabelConfiguration
    pool_size: int
    name: str
    path: str = None
    predictions_path: str = None
