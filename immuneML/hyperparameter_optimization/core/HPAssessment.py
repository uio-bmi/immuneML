import copy
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.core.HPSelection import HPSelection
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.classifiers.SklearnMethod import SklearnMethod
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.MLProcess import MLProcess


class HPAssessment:

    @staticmethod
    def run_assessment(state: TrainMLModelState) -> TrainMLModelState:

        state = HPAssessment._create_root_path(state)
        train_val_datasets, test_datasets = HPUtil.split_data(state.dataset, state.assessment, state.path, state.label_configuration)
        n_splits = len(train_val_datasets)

        for index in range(n_splits):
            state = HPAssessment.run_assessment_split(state, train_val_datasets[index], test_datasets[index], index, n_splits)

        return state

    @staticmethod
    def _create_root_path(state: TrainMLModelState) -> TrainMLModelState:
        name = state.name if state.name is not None else "result"
        state.path = state.path / name
        return state

    @staticmethod
    def run_assessment_split(state, train_val_dataset, test_dataset, split_index: int, n_splits):
        """run inner CV loop (selection) and retrain on the full train_val_dataset after optimal model is chosen"""

        print_log(f'Training ML model: running outer CV loop: started split {split_index + 1}/{n_splits}.\n', include_datetime=True)

        current_path = HPAssessment.create_assessment_path(state, split_index)

        assessment_state = HPAssessmentState(split_index, train_val_dataset, test_dataset, current_path, state.label_configuration)
        state.assessment_states.append(assessment_state)

        state = HPSelection.run_selection(state, train_val_dataset, current_path, split_index)
        state = HPAssessment.run_assessment_split_per_label(state, split_index)

        assessment_state.train_val_data_reports = ReportUtil.run_data_reports(train_val_dataset, state.assessment.reports.data_split_reports.values(),
                                                                              current_path / "data_report_train", state.number_of_processes,
                                                                              state.context)
        assessment_state.test_data_reports = ReportUtil.run_data_reports(test_dataset, state.assessment.reports.data_split_reports.values(),
                                                                         current_path / "data_report_test", state.number_of_processes, state.context)

        print_log(f'Training ML model: running outer CV loop: finished split {split_index + 1}/{n_splits}.\n', include_datetime=True)

        return state

    @staticmethod
    def run_assessment_split_per_label(state: TrainMLModelState, split_index: int):
        """iterate through labels and hp_settings and retrain all models"""
        n_labels = state.label_configuration.get_label_count()

        for idx, label in enumerate(state.label_configuration.get_label_objects()):

            print_log(f"Training ML model: running the inner loop of nested CV: "
                      f"retrain models for label {label.name} (label {idx + 1} / {n_labels}).\n", include_datetime=True)

            path = state.assessment_states[split_index].path

            for index, hp_setting in enumerate(state.hp_settings):

                if hp_setting != state.assessment_states[split_index].label_states[label.name].optimal_hp_setting:
                    setting_path = path / f"{label.name}_{hp_setting}/"
                else:
                    setting_path = path / f"{label.name}_{hp_setting}_optimal/"

                train_val_dataset = state.assessment_states[split_index].train_val_dataset
                test_dataset = state.assessment_states[split_index].test_dataset
                state = HPAssessment.reeval_on_assessment_split(state, train_val_dataset, test_dataset, hp_setting, setting_path, label, split_index)

            print_log(f"Training ML model: running the inner loop of nested CV: completed retraining models "
                      f"for label {label.name} (label {idx + 1} / {n_labels}).\n", include_datetime=True)

        return state

    @staticmethod
    def reeval_on_assessment_split(state, train_val_dataset: Dataset, test_dataset: Dataset, hp_setting: HPSetting, path: Path, label: Label,
                                   split_index: int) -> MLMethod:
        """retrain model for specific label, assessment split and hp_setting"""

        updated_hp_setting = HPAssessment.update_hp_setting_for_assessment(hp_setting, state, split_index, label.name)

        assessment_item = MLProcess(train_dataset=train_val_dataset, test_dataset=test_dataset, label=label, metrics=state.metrics,
                                    optimization_metric=state.optimization_metric, path=path, hp_setting=updated_hp_setting,
                                    report_context=state.context, ml_reports=state.assessment.reports.model_reports.values(),
                                    number_of_processes=state.number_of_processes,
                                    encoding_reports=state.assessment.reports.encoding_reports.values(),
                                    label_config=LabelConfiguration([label]), example_weighting=state.example_weighting).run(split_index)

        state.assessment_states[split_index].label_states[label.name].assessment_items[str(hp_setting)] = assessment_item

        return state

    @staticmethod
    def update_hp_setting_for_assessment(hp_setting: HPSetting, state: TrainMLModelState, split_index: int, label_name: str):

        if isinstance(hp_setting.ml_method, SklearnMethod) and hp_setting.ml_params['model_selection_cv']:
            updated_hp_setting = copy.deepcopy(hp_setting)
            updated_hp_setting.ml_params['model_selection_cv'] = False
            updated_hp_setting.ml_params['model_selection_n_folds'] = -1

            comp_func = ClassificationMetric.get_search_criterion(state.optimization_metric)
            hp_items = state.assessment_states[split_index].label_states[label_name].selection_state.hp_items[hp_setting.get_key()]

            if len(hp_items) > 1:
                optimal_params = {hp_item.performance[state.optimization_metric.name.lower()]:
                                      HPAssessment._get_only_hyperparams(hp_item.method.get_params())
                                  for hp_item in hp_items}
                updated_hp_setting.ml_params[updated_hp_setting.ml_method.__class__.__name__] = optimal_params[
                    comp_func(optimal_params.keys())]

            elif len(hp_items) == 1:
                updated_hp_setting.ml_params[updated_hp_setting.ml_method.__class__.__name__] = hp_items[0].method.model.get_params()

            return updated_hp_setting

        else:
            return hp_setting

    @staticmethod
    def _get_only_hyperparams(params: dict):
        return copy.deepcopy({k: v for k, v in params.items() if k not in ['intercept', 'coefficients']})

    @staticmethod
    def create_assessment_path(state, split_index):
        current_path = state.path / f"split_{split_index + 1}"
        PathBuilder.build(current_path)
        return current_path
