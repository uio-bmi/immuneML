from typing import List

from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.core.HPAssessment import HPAssessment
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from source.reports.data_reports.DataReport import DataReport
from source.workflows.instructions.Instruction import Instruction


class HPOptimizationInstruction(Instruction):
    """
    Class implementing hyper-parameter optimization and nested model training and assessment:

    The process is defined by two loops:
        - the outer loop over defined splits of the dataset for performance assessment
        - the inner loop over defined hyper-parameter space and with cross-validation or train & validation split
          to choose the best hyper-parameters

    Optimal model chosen by the inner loop is then retrained on the whole training dataset in the outer loop.

    """

    def __init__(self, dataset, hp_strategy: HPOptimizationStrategy, hp_settings: list, assessment: SplitConfig, selection: SplitConfig,
                 metrics: set, optimization_metric: MetricType, label_configuration: LabelConfiguration, path: str = None,
                 context: dict = None, batch_size: int = 1, data_reports: List[DataReport] = None, name: str = None):
        self.hp_optimization_state = HPOptimizationState(dataset, hp_strategy, hp_settings, assessment, selection, metrics,
                                                         optimization_metric, label_configuration, path, context, batch_size, data_reports,
                                                         name)

    def run(self, result_path: str):
        self.hp_optimization_state.path = result_path
        state = HPAssessment.run_assessment(self.hp_optimization_state)
        self.print_performances(state)
        return state

    def print_performances(self, state: HPOptimizationState):
        print("Performances -----------------------------------------------")

        for label in state.label_configuration.get_labels_by_name():
            print(f"\n\nLabel: {label}")
            print("Performance per assessment split:")
            for split in range(state.assessment.split_count):
                print(f"Split {split+1}: {state.assessment_states[split].label_states[label].optimal_assessment_item.performance}")
            print(f"Average performance: "
                  f"{sum([state.assessment_states[split].label_states[label].optimal_assessment_item.performance for split in range(state.assessment.split_count)])/state.assessment.split_count}")
            print("------------------------------")
