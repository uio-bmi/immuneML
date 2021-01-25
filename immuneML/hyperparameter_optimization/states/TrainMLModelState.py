from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Dict

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Metric import Metric
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from immuneML.reports.ReportResult import ReportResult


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
    path: Path = None
    context: dict = None
    number_of_processes: int = 1
    reports: dict = field(default_factory=dict)
    name: str = None
    refit_optimal_model: bool = None
    store_encoded_data: bool = None
    optimal_hp_items: Dict[str, HPItem] = field(default_factory=dict)
    optimal_hp_item_paths: Dict[str, str] = field(default_factory=dict)
    assessment_states: List[HPAssessmentState] = field(default_factory=list)
    report_results: List[ReportResult] = field(default_factory=list)
