import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union, Any

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.SampleConfig import SampleConfig
from immuneML.reports.ReportResult import ReportResult
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem, ClusteringSetting



@dataclass
class StabilityConfig:
    split_count: int = None
    random_seed: int = None


@dataclass
class ClusteringConfig:
    name: str
    dataset: Dataset
    metrics: List[str]
    sample_config: SampleConfig
    stability_config: StabilityConfig
    clustering_settings: List[ClusteringSetting]
    region_type: RegionType = RegionType.IMGT_CDR3
    label_config: LabelConfiguration = None
    sequence_type: SequenceType = SequenceType.AMINO_ACID
    random_labeling_count: int = None

    def get_cl_setting_by_key(self, key: str) -> ClusteringSetting:
        cl_setting = [cs for cs in self.clustering_settings if cs.get_key() == key]
        if cl_setting and len(cl_setting) == 1:
            return cl_setting[0]
        else:
            logging.warning(f"No cluster setting for key {key}. "
                            f"Available settings: {[cs.get_key() for cs in self.clustering_settings]}")
            return None


@dataclass
class ClusteringItemResult:
    item: ClusteringItem
    report_results: List[ReportResult] = field(default_factory=list)


@dataclass
class ClusteringResultPerRun:
    run_id: int
    items: Dict[str, ClusteringItemResult] = field(default_factory=dict)

    def get_cl_item(self, cl_setting: Union[str, ClusteringSetting]):
        key = cl_setting if isinstance(cl_setting, str) else cl_setting.get_key()
        return self.items[key].item


@dataclass
class ClusteringState:
    name: str
    config: ClusteringConfig
    result_path: Path = None
    clustering_items: List[ClusteringResultPerRun] = field(default_factory=list)
    predictions_paths: List[Path] = None
    subsampled_datasets: List[Dataset] = None
    clustering_report_results: List[ReportResult] = field(default_factory=list)
    metrics_performance_paths: Dict[str, Path] = field(default_factory=dict)
    optimal_settings_on_discovery: Dict[str, ClusteringItemResult] = field(default_factory=dict)
    final_predictions_path: Path = None
    best_settings_zip_paths: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_cl_result_per_run(self, run_id: int, cl_item_result: ClusteringResultPerRun):
        if len(self.clustering_items) <= run_id:
            self.clustering_items.append(cl_item_result)
        else:
            self.clustering_items[run_id] = cl_item_result
