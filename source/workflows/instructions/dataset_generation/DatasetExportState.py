from dataclasses import dataclass
from typing import List

from source.data_model.dataset.Dataset import Dataset


@dataclass
class DatasetExportState:
    datasets: List[Dataset]
    formats: List[str]
    paths: dict
    result_path: str
    name: str
