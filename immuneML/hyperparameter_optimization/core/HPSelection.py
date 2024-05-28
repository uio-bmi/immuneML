from pathlib import Path

from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.MLProcess import MLProcess


class HPSelection:

    @staticmethod
    def update_split_count(state: TrainMLModelState, train_val_dataset):
        if state.selection.split_strategy == SplitType.LOOCV:
            state.selection.split_count = train_val_dataset.get_example_count()

        return state

    @staticmethod
    def run_selection(state: TrainMLModelState, train_val_dataset, current_path: Path, split_index: int) -> TrainMLModelState:

        path = HPSelection.create_selection_path(state, current_path)
        state = HPSelection.update_split_count(state, train_val_dataset)
        train_datasets, val_datasets = HPUtil.split_data(train_val_dataset, state.selection, path, state.label_configuration)

        n_labels = state.label_configuration.get_label_count()

        for idx, label in enumerate(state.label_configuration.get_label_objects()):

            print_log(f"Hyperparameter optimization: running the inner loop of nested CV: selection for label {label.name} "
                      f"(label {idx + 1} / {n_labels}).\n", include_datetime=True)

            selection_state = HPSelectionState(train_datasets, val_datasets, path, state.hp_strategy)
            state.assessment_states[split_index].label_states[label.name].selection_state = selection_state

            hp_setting = selection_state.hp_strategy.generate_next_setting()
            while hp_setting is not None:
                performance = HPSelection.evaluate_hp_setting(state, hp_setting, train_datasets, val_datasets,
                                                              path, label, split_index)
                hp_setting = selection_state.hp_strategy.generate_next_setting(hp_setting, performance)

            HPUtil.run_selection_reports(state, train_val_dataset, train_datasets, val_datasets, selection_state)

            print_log(f"Hyperparameter optimization: running the inner loop of nested CV: completed selection for "
                      f"label {label.name} (label {idx + 1} / {n_labels}).\n", include_datetime=True)

        return state

    @staticmethod
    def evaluate_hp_setting(state: TrainMLModelState, hp_setting: HPSetting, train_datasets: list, val_datasets: list,
                            current_path: Path, label: Label, assessment_split_index: int):

        performances = []
        for index in range(state.selection.split_count):
            performance = HPSelection.run_setting(state, hp_setting, train_datasets[index], val_datasets[index], index + 1,
                                                  current_path / f"split_{index + 1}" / f"{label.name}_{hp_setting.get_key()}",
                                                  label, assessment_split_index)
            performances.append(performance)

        return HPUtil.get_average_performance(performances)

    @staticmethod
    def run_setting(state: TrainMLModelState, hp_setting, train_dataset, val_dataset, split_index: int,
                    current_path: Path, label: Label, assessment_index: int):

        hp_item = MLProcess(train_dataset=train_dataset, test_dataset=val_dataset, encoding_reports=state.selection.reports.encoding_reports.values(),
                            label_config=LabelConfiguration([label]), report_context=state.context,
                            number_of_processes=state.number_of_processes, metrics=state.metrics, optimization_metric=state.optimization_metric,
                            ml_reports=state.selection.reports.model_reports.values(), label=label, path=current_path, hp_setting=hp_setting,
                            example_weighting=state.example_weighting).run(split_index)

        state.assessment_states[assessment_index].label_states[label.name].selection_state.hp_items[hp_setting.get_key()].append(hp_item)

        return hp_item.performance[state.optimization_metric.name.lower()] if hp_item.performance is not None else None

    @staticmethod
    def create_selection_path(state: TrainMLModelState, current_path: Path) -> Path:
        path = current_path / f"selection_{state.selection.split_strategy.name.lower()}"
        PathBuilder.build(path)
        return path
