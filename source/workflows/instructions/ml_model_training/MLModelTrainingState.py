from dataclasses import dataclass, field
from typing import List, Dict

from source.hyperparameter_optimization.states.HPItem import HPItem
from source.workflows.instructions.MLProcess import MLProcess


@dataclass
class MLModelTrainingState:

    processes: List[MLProcess] = None
    result_path: str = None
    name: str = None
    hp_items: Dict[str, HPItem] = field(default_factory=dict)
