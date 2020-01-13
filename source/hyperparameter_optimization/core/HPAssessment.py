from source.data_model.dataset.Dataset import Dataset
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.core.HPReports import HPReports
from source.hyperparameter_optimization.core.HPSelection import HPSelection
from source.hyperparameter_optimization.core.HPUtil import HPUtil
from source.hyperparameter_optimization.states.HPAssessmentItem import HPAssessmentItem
from source.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder


class HPAssessment:

    @staticmethod
    def run_assessment(state: HPOptimizationState) -> HPOptimizationState:
        train_val_datasets, test_datasets = HPUtil.split_data(state.dataset, state.assessment_config, state.path)

        for index in range(len(train_val_datasets)):
            state = HPAssessment.run_assessment_split(state, train_val_datasets[index], test_datasets[index], index)

        HPReports.run_hyperparameter_reports(state, f"{state.path}hyperparameter_reports/")

        return state

    @staticmethod
    def run_assessment_split(state, train_val_dataset, test_dataset, split_index: int):

        current_path = HPAssessment.create_assessment_path(state, split_index)

        state.assessment_states.append(HPAssessmentState(split_index, train_val_dataset, test_dataset, current_path, state.label_configuration))

        state = HPSelection.run_selection(state, train_val_dataset, current_path, split_index)

        state = HPAssessment.run_assessment_split_per_label(state, split_index)

        state = HPAssessment.assess_split_performance(state, train_val_dataset=train_val_dataset, test_dataset=test_dataset,
                                                      hp_settings=state.hp_settings, run=split_index, path=current_path)

        return state

    @staticmethod
    def run_assessment_split_per_label(state: HPOptimizationState, split_index: int):

        for label in state.label_configuration.get_labels_by_name():
            state = HPAssessment.run_assessment_split_for_label(state, label, split_index, f"{state.assessment_states[split_index].path}{label}/")

        return state

    @staticmethod
    def run_assessment_split_for_label(state: HPOptimizationState, label, split_index: int, path) -> HPOptimizationState:
        for index, hp_setting in enumerate(state.hp_settings):

            method = HPAssessment.retrain_on_assessment_split(state, state.assessment_states[split_index].train_val_dataset, hp_setting,
                                                              f"{path}{hp_setting}/")

            assessment_item = HPAssessmentItem(method=method, hp_setting=hp_setting)
            state.assessment_states[split_index].label_states[label].assessment_items[hp_setting] = assessment_item

        return state

    @staticmethod
    def retrain_on_assessment_split(state, train_val_dataset: Dataset, hp_setting: HPSetting, path: str) -> MLMethod:
        PathBuilder.build(path)

        processed_dataset = HPUtil.preprocess_dataset(train_val_dataset, hp_setting.preproc_sequence, path)
        encoded_train_dataset = HPUtil.encode_dataset(processed_dataset, hp_setting, path, learn_model=True, context=state.context,
                                                      batch_size=state.batch_size, label_configuration=state.label_configuration)
        method = HPUtil.train_method(state, encoded_train_dataset, hp_setting, path)

        return method

    @staticmethod
    def assess_split_performance(state: HPOptimizationState, train_val_dataset: Dataset, test_dataset: Dataset, hp_settings: list,
                                 run: int, path: str):

        if test_dataset.get_example_count() > 0:
            PathBuilder.build(path)
            assessment_performances = []
            performance = None

            for hp_setting in hp_settings:
                for label in state.label_configuration.get_labels_by_name():

                    tmp_path = f"{path}{label}/{hp_setting}/"
                    processed_test_dataset = HPUtil.preprocess_dataset(test_dataset, hp_setting.preproc_sequence, tmp_path)
                    encoded_test_dataset = HPUtil.encode_dataset(processed_test_dataset, hp_setting, tmp_path, learn_model=False,
                                                                 context=state.context, batch_size=state.batch_size,
                                                                 label_configuration=state.label_configuration)

                    hp_performance = HPUtil.assess_performance(state, encoded_test_dataset, run, path, hp_setting)
                    state.assessment_states[run].label_states[label].assessment_items[hp_setting].performance = hp_performance[label]

                HPReports.run_assessment_reports(state, f"{path}/reports", run)
                assessment_performances.append(performance)

        return state

    @staticmethod
    def create_assessment_path(state, run):
        current_path = f"{state.path}assessment_{state.assessment_config.split_strategy.name}/run_{run+1}/"
        PathBuilder.build(current_path)
        return current_path
