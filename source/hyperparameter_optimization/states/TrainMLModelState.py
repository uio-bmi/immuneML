from dataclasses import dataclass, field
from typing import List, Set, Dict

from source.data_model.dataset.Dataset import Dataset
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from source.hyperparameter_optimization.states.HPItem import HPItem
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from source.reports.ReportResult import ReportResult


@dataclass
class TrainMLModelState:
    dataset: Dataset
    hp_strategy: HPOptimizationStrategy
    hp_settings: List[HPSetting]
    assessment: SplitConfig
    selection: SplitConfig
    metrics: Set[Metric]
    optimization_metric: Metric
    label_configuration: LabelConfiguration
    path: str = None
    context: dict = None
    batch_size: int = 1
    data_reports: dict = field(default_factory=dict)
    name: str = None
    refit_optimal_model: bool = None
    store_encoded_data: bool = None
    optimal_hp_items: Dict[str, HPItem] = field(default_factory=dict)
    optimal_hp_item_paths: Dict[str, str] = field(default_factory=dict)
    assessment_states: List[HPAssessmentState] = field(default_factory=list)
    hp_report_results: List[ReportResult] = field(default_factory=list)
    data_report_results: List[ReportResult] = field(default_factory=list)
