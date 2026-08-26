from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.Label import Label


@dataclass
class SubsamplingState:

    dataset: Dataset
    subsampled_dataset_sizes: List[int] = field(default_factory=list)
    subsampled_repertoire_size: int = None
    label: Label = None
    subsampled_class_distributions: List[dict] = None
    result_path: Path = None
    name: str = None
    subsampled_datasets: List[Dataset] = field(default_factory=list)
    subsampled_dataset_paths: dict = field(default_factory=dict)
