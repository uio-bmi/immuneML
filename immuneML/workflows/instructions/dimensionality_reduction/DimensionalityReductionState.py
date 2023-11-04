from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset


@dataclass
class DimensionalityReductionState:
    name: str
    result_path: Path = None
