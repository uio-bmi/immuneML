from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset


@dataclass
class DatasetExportState:
    datasets: List[Dataset]
    formats: List[str]
    paths: dict
    result_path: Path
    name: str
