from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.reports.ReportResult import ReportResult
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem, ClusteringSetting


@dataclass
class ClusteringConfig:
    name: str
    dataset: Dataset
    metrics: List[str]
    split_config: SplitConfig
    validation_type: List[str]
    clustering_settings: List[ClusteringSetting]
    region_type: RegionType = RegionType.IMGT_CDR3
    label_config: LabelConfiguration = None
    sequence_type: SequenceType = SequenceType.AMINO_ACID


@dataclass
class ClusteringState:
    name: str
    config: ClusteringConfig
    result_path: Path = None
    clustering_items: List[Dict[str, Dict[str, ClusteringItem]]] = field(default_factory=list)
    predictions_paths: List[Dict[str, Path]] = None
    discovery_datasets: List[Dataset] = None
    validation_datasets: List[Dataset] = None
    cl_item_report_results: List[Dict[str, Dict[str, List[ReportResult]]]] = None
    clustering_report_results: List[ReportResult] = field(default_factory=list)
