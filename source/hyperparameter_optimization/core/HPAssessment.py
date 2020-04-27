from source.data_model.dataset.Dataset import Dataset
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.core.HPReports import HPReports
from source.hyperparameter_optimization.core.HPSelection import HPSelection
from source.hyperparameter_optimization.core.HPUtil import HPUtil
from source.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from source.hyperparameter_optimization.states.HPItem import HPItem
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder


class HPAssessment:

    @staticmethod
    def run_assessment(state: HPOptimizationState) -> HPOptimizationState:

        state = HPAssessment._create_root_path(state)
        train_val_datasets, test_datasets = HPUtil.split_data(state.dataset, state.assessment, state.path)

        for index in range(len(train_val_datasets)):
            state = HPAssessment.run_assessment_split(state, train_val_datasets[index], test_datasets[index], index)

        HPReports.run_hyperparameter_reports(state, f"{state.path}hyperparameter_reports/")
        HPReports.run_data_reports(state, f"{state.path}data_reports/")

        return state

    @staticmethod
    def _create_root_path(state: HPOptimizationState) -> HPOptimizationState:
        state.path = f"{state.path}assessment_{state.assessment.split_strategy.name.lower()}/"
        return state

    @staticmethod
    def run_assessment_split(state, train_val_dataset, test_dataset, split_index: int):

        current_path = HPAssessment.create_assessment_path(state, split_index)

        assessment_state = HPAssessmentState(split_index, train_val_dataset, test_dataset, current_path, state.label_configuration)
        state.assessment_states.append(assessment_state)

        state = HPSelection.run_selection(state, train_val_dataset, current_path, split_index)

        state = HPAssessment.run_assessment_split_per_label(state, split_index)

        return state

    @staticmethod
    def run_assessment_split_per_label(state: HPOptimizationState, split_index: int):

        for label in state.label_configuration.get_labels_by_name():
            state = HPAssessment.run_assessment_split_for_label(state, label, split_index, f"{state.assessment_states[split_index].path}")

        HPReports.run_assessment_reports(state, state.assessment_states[split_index].path, split_index)

        return state

    @staticmethod
    def run_assessment_split_for_label(state: HPOptimizationState, label, split_index: int, path) -> HPOptimizationState:
        for index, hp_setting in enumerate(state.hp_settings):

            if hp_setting != state.assessment_states[split_index].label_states[label].optimal_hp_setting:
                setting_path = f"{path}{label}_{hp_setting}/"
            else:
                setting_path = f"{path}{label}_{hp_setting}_optimal/"
            train_predictions_path = f"{setting_path}train_predictions.csv"
            test_predictions_path = f"{setting_path}test_predictions.csv"
            ml_details_path = f"{setting_path}ml_details.yaml"
            ml_score_path = f"{setting_path}ml_score.csv"

            train_val_dataset = state.assessment_states[split_index].train_val_dataset
            test_dataset = state.assessment_states[split_index].test_dataset
            state = HPAssessment.reeval_on_assessment_split(state, train_val_dataset, test_dataset, hp_setting, setting_path,
                                                            train_predictions_path, ml_details_path, test_predictions_path, label,
                                                            split_index, ml_score_path)

        return state

    @staticmethod
    def reeval_on_assessment_split(state, train_val_dataset: Dataset, test_dataset: Dataset, hp_setting: HPSetting, path: str,
                                   train_predictions_path: str, ml_details_path: str, test_predictions_path: str, label: str,
                                   split_index: int, ml_score_path: str) -> MLMethod:
        PathBuilder.build(path)

        processed_dataset = HPUtil.preprocess_dataset(train_val_dataset, hp_setting.preproc_sequence, path)
        encoded_train_dataset = HPUtil.encode_dataset(processed_dataset, hp_setting, path, learn_model=True, context=state.context,
                                                      batch_size=state.batch_size, label_configuration=state.label_configuration)
        method = HPUtil.train_method(label, encoded_train_dataset, hp_setting, path, train_predictions_path, ml_details_path)

        processed_test_dataset = HPUtil.preprocess_dataset(test_dataset, hp_setting.preproc_sequence, path)
        encoded_test_dataset = HPUtil.encode_dataset(processed_test_dataset, hp_setting, path, learn_model=False, context=state.context,
                                                     batch_size=state.batch_size, label_configuration=state.label_configuration)

        assessment_item = HPItem(method=method, hp_setting=hp_setting, train_predictions_path=train_predictions_path,
                                 test_predictions_path=test_predictions_path, ml_details_path=ml_details_path,
                                 train_dataset=train_val_dataset, test_dataset=test_dataset, split_index=split_index)

        state.assessment_states[split_index].label_states[label].assessment_items[hp_setting] = assessment_item

        performance = HPUtil.assess_performance(state, encoded_test_dataset, split_index, path, hp_setting, test_predictions_path, label, ml_score_path)
        assessment_item.performance = performance

        return state

    @staticmethod
    def create_assessment_path(state, split_index):
        current_path = f"{state.path}split_{split_index+1}/"
        PathBuilder.build(current_path)
        return current_path
