import copy

from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from source.reports.ReportUtil import ReportUtil


class HPReports:

    @staticmethod
    def run_data_reports(state: HPOptimizationState, path: str):
        if state.data_reports is not None:
            state.data_report_results = ReportUtil.run_data_reports(state.dataset, state.data_reports.values(), path, state.context)

    @staticmethod
    def run_hyperparameter_reports(state: HPOptimizationState, path: str):
        for key, report in state.assessment.reports.hyperparameter_reports.items():
            tmp_report = copy.deepcopy(report)
            tmp_report.hp_optimization_state = state
            tmp_report.result_path = f"{path}{key}/"
            report_result = tmp_report.generate_report()
            state.hp_report_results.append(report_result)

    @staticmethod
    def run_assessment_reports(state: HPOptimizationState, path: str, split_index: int):
        train_val_dataset = state.assessment_states[split_index].train_val_dataset
        test_dataset = state.assessment_states[split_index].test_dataset

        # data_split reports
        data_split_reports = state.assessment.reports.data_split_reports.values()
        state.assessment_states[split_index].train_val_data_reports = ReportUtil.run_data_reports(train_val_dataset, data_split_reports,
                                                                                                  f"{path}/reports/train/", state.context)
        state.assessment_states[split_index].test_data_reports.append(ReportUtil.run_data_reports(test_dataset, data_split_reports,
                                                                                                  f"{path}/reports/test/", state.context))

        # ML model reports
        optimal_model_reports = state.assessment.reports.optimal_model_reports.values()
        for label in state.label_configuration.get_labels_by_name():
            opt_assesment_item = state.assessment_states[split_index].label_states[label].optimal_assessment_item
            opt_assesment_item.model_report_results += ReportUtil.run_ML_reports(train_val_dataset, test_dataset, opt_assesment_item.method,
                                                                                 optimal_model_reports, f"{path}reports/{label}/optimal_model/",
                                                                                 opt_assesment_item.hp_setting, label, state.context)

    @staticmethod
    def run_selection_reports(state: HPOptimizationState, dataset, train_datasets: list, val_datasets: list, selection_state: HPSelectionState):
        path = selection_state.path
        data_split_reports = state.selection.reports.data_split_reports.values()
        for index in range(len(train_datasets)):
            selection_state.train_data_reports = ReportUtil.run_data_reports(train_datasets[index], data_split_reports,
                                                                             path + f"split_{index+1}/reports/train/", state.context)
            selection_state.val_data_reports = ReportUtil.run_data_reports(val_datasets[index], data_split_reports,
                                                                           path + f"split_{index+1}/reports/test/", state.context)

        data_reports = state.selection.reports.data_reports.values()
        selection_state.data_reports = ReportUtil.run_data_reports(dataset, data_reports, f"{path}reports/", state.context)

