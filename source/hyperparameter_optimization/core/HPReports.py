import copy

from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.ml_methods.MLMethod import MLMethod
from source.reports.data_reports.DataReport import DataReport
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class HPReports:

    @staticmethod
    def run_hyperparameter_reports(state: HPOptimizationState, path: str):
        PathBuilder.build(path)

        for report in state.assessment_config.reports.hyperparameter_reports:
            tmp_report = copy.deepcopy(report)
            tmp_report.hp_optimization_state = state
            tmp_report.path = path
            tmp_report.generate_report()

    @staticmethod
    def run_assessment_reports(state: HPOptimizationState, path: str, split_index: int):
        train_val_dataset = state.assessment_states[split_index].train_val_dataset
        test_dataset = state.assessment_states[split_index].test_dataset

        for report in state.assessment_config.reports.data_split_reports:
            HPReports.run_data_report(state, report, train_val_dataset, path + "reports/train/")
            HPReports.run_data_report(state, report, test_dataset, path + "reports/test/")

        for report in state.assessment_config.reports.optimal_model_reports:
            for label in state.label_configuration.get_labels_by_name():
                method = state.assessment_states[split_index].label_states[label].optimal_assessment_item.method
                HPReports.run_model_report(state, report, train_val_dataset, test_dataset, method, f"{path}reports/label_{label}/")

    @staticmethod
    def run_selection_reports(state: HPOptimizationState, dataset, train_datasets: list, val_datasets: list, path: str):

        for report in state.selection_config.reports.data_split_reports:
            for index in range(len(train_datasets)):
                HPReports.run_data_report(state, report, train_datasets[index], path + "split_{}/reports/train/".format(index + 1))
                HPReports.run_data_report(state, report, val_datasets[index], path + "split_{}/reports/test/".format(index + 1))

        for report in state.selection_config.reports.data_reports:
            HPReports.run_data_report(state, report, dataset, path + "reports/")

    @staticmethod
    def run_model_report(state: HPOptimizationState, report: MLReport, train_dataset, test_dataset, method: MLMethod, path: str):
        tmp_report = copy.deepcopy(report)
        tmp_report.train_dataset = train_dataset
        tmp_report.test_dataset = test_dataset
        tmp_report.method = method
        tmp_report.path = path
        tmp_report.set_context(state.context)
        tmp_report.generate_report()

    @staticmethod
    def run_data_report(state: HPOptimizationState, report: DataReport, dataset, path: str):
        tmp_report = copy.deepcopy(report)
        tmp_report.dataset = dataset
        tmp_report.result_path = path
        tmp_report.set_context(state.context)
        tmp_report.generate_report()
