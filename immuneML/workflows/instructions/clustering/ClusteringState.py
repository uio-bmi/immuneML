from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

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
class ClusteringItemResult:
    item: ClusteringItem
    report_results: List[ReportResult] = field(default_factory=list)


@dataclass
class ClusteringResultPerRun:
    run_id: int
    run_type: str
    items: Dict[str, ClusteringItemResult] = field(default_factory=dict)

    def get_cl_item(self, cl_setting: Union[str, ClusteringSetting]):
        key = cl_setting if isinstance(cl_setting, str) else cl_setting.get_key()
        return self.items[key].item


@dataclass
class ClusteringResults:
    discovery: ClusteringResultPerRun = None
    method_based_validation: ClusteringResultPerRun = None
    result_based_validation: ClusteringResultPerRun = None


@dataclass
class ClusteringState:
    name: str
    config: ClusteringConfig
    result_path: Path = None
    clustering_items: List[ClusteringResults] = field(default_factory=list)
    predictions_paths: List[Dict[str, Path]] = None
    discovery_datasets: List[Dataset] = None
    validation_datasets: List[Dataset] = None
    clustering_report_results: List[ReportResult] = field(default_factory=list)

    def add_cl_result_per_run(self, run_id: int, analysis_desc: str, cl_item_result: ClusteringResultPerRun):
        if len(self.clustering_items) <= run_id:
            self.clustering_items.append(ClusteringResults(**{analysis_desc: cl_item_result}))
        else:
            setattr(self.clustering_items[run_id], analysis_desc, cl_item_result)
