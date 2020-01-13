from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.core.HPReports import HPReports
from source.hyperparameter_optimization.core.HPUtil import HPUtil
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from source.util.PathBuilder import PathBuilder
from source.workflows.processes.MLProcess import MLProcess


class HPSelection:

    @staticmethod
    def run_selection(state: HPOptimizationState, train_val_dataset, current_path: str, split_index: int) -> HPOptimizationState:
        for label in state.label_configuration.get_labels_by_name():
            path = HPSelection.create_selection_path(state, current_path, label)

            train_datasets, val_datasets = HPUtil.split_data(train_val_dataset, state.selection_config, path)
            selection_state = HPSelectionState(train_datasets, val_datasets, path, state.hp_strategy)

            hp_setting = selection_state.hp_strategy.generate_next_setting()
            while hp_setting is not None:
                performance = HPSelection.evaluate_hp_setting(state, hp_setting, train_datasets, val_datasets, selection_state.path, label)
                hp_setting = selection_state.hp_strategy.generate_next_setting(hp_setting, performance)

            HPReports.run_selection_reports(state, train_val_dataset, train_datasets, val_datasets, selection_state.path + "reports/")

            state.assessment_states[split_index].label_states[label].selection_state = selection_state

        return state

    @staticmethod
    def evaluate_hp_setting(state: HPOptimizationState, hp_setting: HPSetting, train_datasets: list, val_datasets: list,
                            current_path: str, label: str) -> float:

        performances = []
        for index in range(state.selection_config.split_count):
            performance = HPSelection.run_setting(state, hp_setting, train_datasets[index], val_datasets[index], index + 1, current_path,
                                                  label)
            performances.append(performance)

        if all(performance is not None for performance in performances):
            return HPUtil.get_average_performance(performances, label)
        else:
            return -1.

    @staticmethod
    def run_setting(state: HPOptimizationState, hp_setting, train_dataset, val_dataset, split_index: int, current_path: str, label: str):
        path = HPSelection.create_setting_path(current_path, hp_setting, split_index)

        new_train_dataset = HPUtil.preprocess_dataset(train_dataset, hp_setting.preproc_sequence, path + "train/")
        new_val_dataset = HPUtil.preprocess_dataset(val_dataset, hp_setting.preproc_sequence, path + "val/")

        ml_process = MLProcess(train_dataset=new_train_dataset, test_dataset=new_val_dataset,
                               label_configuration=LabelConfiguration([state.label_configuration.get_label_object(label)]),
                               batch_size=state.batch_size,
                               encoder=hp_setting.encoder.create_encoder(train_dataset, hp_setting.encoder_params).set_context(state.context),
                               encoder_params=hp_setting.encoder_params, method=hp_setting.ml_method,
                               ml_params=hp_setting.ml_params, metrics=state.metrics, path=path,
                               reports=[report.set_context(state.context) for report in state.selection_config.reports.model_reports])
        performance = ml_process.run(split_index)

        return performance

    @staticmethod
    def create_setting_path(current_path: str, hp_setting: HPSetting, run: int):
        path = current_path + "{}/split_{}/".format(hp_setting, run)
        PathBuilder.build(path)
        return path

    @staticmethod
    def create_selection_path(state: HPOptimizationState, current_path: str, label: str) -> str:
        path = "{}{}/selection_{}/".format(current_path, label, state.selection_config.split_strategy.name)
        PathBuilder.build(path)
        return path
