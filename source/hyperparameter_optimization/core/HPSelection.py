import datetime

from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.core.HPUtil import HPUtil
from source.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from source.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.MLProcess import MLProcess


class HPSelection:

    @staticmethod
    def update_split_count(state: TrainMLModelState, train_val_dataset):
        if state.selection.split_strategy == SplitType.LOOCV:
            state.selection.split_count = train_val_dataset.get_example_count()

        return state

    @staticmethod
    def run_selection(state: TrainMLModelState, train_val_dataset, current_path: str, split_index: int) -> TrainMLModelState:

        path = HPSelection.create_selection_path(state, current_path)
        state = HPSelection.update_split_count(state, train_val_dataset)
        train_datasets, val_datasets = HPUtil.split_data(train_val_dataset, state.selection, path)

        n_labels = state.label_configuration.get_label_count()

        for idx, label in enumerate(state.label_configuration.get_labels_by_name()):

            print(f"{datetime.datetime.now()}: Hyperparameter optimization: running the inner loop of nested CV: selection for label {label} "
                  f"(label {idx + 1} / {n_labels}).\n", flush=True)

            selection_state = HPSelectionState(train_datasets, val_datasets, path, state.hp_strategy)
            state.assessment_states[split_index].label_states[label].selection_state = selection_state

            hp_setting = selection_state.hp_strategy.generate_next_setting()
            while hp_setting is not None:
                performance = HPSelection.evaluate_hp_setting(state, hp_setting, train_datasets, val_datasets,
                                                              path, label, split_index)
                hp_setting = selection_state.hp_strategy.generate_next_setting(hp_setting, performance)

            HPUtil.run_selection_reports(state, train_val_dataset, train_datasets, val_datasets, selection_state)

            print(f"{datetime.datetime.now()}: Hyperparameter optimization: running the inner loop of nested CV: completed selection for "
                  f"label {label} (label {idx + 1} / {n_labels}).\n", flush=True)

        return state

    @staticmethod
    def evaluate_hp_setting(state: TrainMLModelState, hp_setting: HPSetting, train_datasets: list, val_datasets: list,
                            current_path: str, label: str, assessment_split_index: int) -> float:

        performances = []
        for index in range(state.selection.split_count):
            performance = HPSelection.run_setting(state, hp_setting, train_datasets[index], val_datasets[index], index + 1,
                                                  f"{current_path}split_{index + 1}/{label}_{hp_setting.get_key()}/",
                                                  label, assessment_split_index)
            performances.append(performance)

        if all(performance is not None for performance in performances):
            return HPUtil.get_average_performance(performances)
        else:
            return -1.

    @staticmethod
    def run_setting(state: TrainMLModelState, hp_setting, train_dataset, val_dataset, split_index: int,
                    current_path: str, label: str, assessment_index: int):

        hp_item = MLProcess(train_dataset=train_dataset, test_dataset=val_dataset, encoding_reports=state.selection.reports.encoding_reports.values(),
                            label_config=LabelConfiguration([state.label_configuration.get_label_object(label)]), report_context=state.context,
                            number_of_processes=state.batch_size, metrics=state.metrics, optimization_metric=state.optimization_metric,
                            ml_reports=state.selection.reports.model_reports.values(), label=label, path=current_path, hp_setting=hp_setting,
                            store_encoded_data=state.store_encoded_data)\
            .run(split_index)

        state.assessment_states[assessment_index].label_states[label].selection_state.hp_items[hp_setting.get_key()].append(hp_item)

        return hp_item.performance

    @staticmethod
    def create_selection_path(state: TrainMLModelState, current_path: str) -> str:
        path = "{}selection_{}/".format(current_path, state.selection.split_strategy.name.lower())
        PathBuilder.build(path)
        return path
