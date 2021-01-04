from dataclasses import dataclass
from pathlib import Path

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
    store_encoded_data: bool
    path: Path = None
    predictions_path: Path = None
