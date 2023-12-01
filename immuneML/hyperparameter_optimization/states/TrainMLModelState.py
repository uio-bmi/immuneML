from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Dict

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.reports.ReportResult import ReportResult


@dataclass
class TrainMLModelState:
    dataset: Dataset
    hp_strategy: HPOptimizationStrategy
    hp_settings: List[HPSetting]
    assessment: SplitConfig
    selection: SplitConfig
    metrics: Set[ClassificationMetric]
    optimization_metric: ClassificationMetric
    label_configuration: LabelConfiguration
    path: Path = None
    context: dict = None
    number_of_processes: int = 1
    reports: dict = field(default_factory=dict)
    name: str = None
    refit_optimal_model: bool = None
    export_all_ml_settings: bool = None
    example_weighting: ExampleWeightingStrategy = None
    optimal_hp_items: Dict[str, HPItem] = field(default_factory=dict)
    optimal_hp_item_paths: Dict[str, str] = field(default_factory=dict)
    assessment_states: List[HPAssessmentState] = field(default_factory=list)
    report_results: List[ReportResult] = field(default_factory=list)
