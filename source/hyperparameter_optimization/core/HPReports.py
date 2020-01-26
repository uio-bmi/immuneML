import copy

from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.states import HPItem
from source.ml_methods.MLMethod import MLMethod
from source.reports.data_reports.DataReport import DataReport
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder

class HPReports:

    @staticmethod
    def run_hyperparameter_reports(state: HPOptimizationState, path: str):
        for key, report in state.assessment_config.reports.hyperparameter_reports.items():
            tmp_report = copy.deepcopy(report)
            tmp_report.hp_optimization_state = state
            tmp_report.result_path = f"{path}{key}/"
            tmp_report.generate_report()

    @staticmethod
    def run_assessment_reports(state: HPOptimizationState, path: str, split_index: int):
        train_val_dataset = state.assessment_states[split_index].train_val_dataset
        test_dataset = state.assessment_states[split_index].test_dataset

        for key, report in state.assessment_config.reports.data_split_reports.items():
            HPReports.run_data_report(state, report, train_val_dataset, path + "reports/{}/train/".format(key))
            HPReports.run_data_report(state, report, test_dataset, path + "reports/{}/test/".format(key))


        for key, report in state.assessment_config.reports.optimal_model_reports.items():
            for label in state.label_configuration.get_labels_by_name():
                opt_assesment_item = state.assessment_states[split_index].label_states[label].optimal_assessment_item
                HPReports.run_model_report(state, report, opt_assesment_item, label, f"{path}reports/{key}/label_{label}/")

    @staticmethod
    def run_selection_reports(state: HPOptimizationState, dataset, train_datasets: list, val_datasets: list, path: str):

        for key, report in state.selection_config.reports.data_split_reports.items():
            for index in range(len(train_datasets)):
                HPReports.run_data_report(state, report, train_datasets[index], path + "split_{}/reports/{}/train/".format(index + 1, key))
                HPReports.run_data_report(state, report, val_datasets[index], path + "split_{}/reports/{}/test/".format(index + 1, key))

        for key, report in state.selection_config.reports.data_reports.items():
            HPReports.run_data_report(state, report, dataset, path + "reports/{}/".format(key))

    @staticmethod
    def run_model_report(state: HPOptimizationState, report: MLReport, opt_assesment_item: HPItem, label: str, path: str):
        tmp_report = copy.deepcopy(report)
        tmp_report.train_dataset = opt_assesment_item.train_dataset
        tmp_report.test_dataset = opt_assesment_item.test_dataset
        tmp_report.method = opt_assesment_item.method
        tmp_report.ml_details_path = opt_assesment_item.ml_details_path # Only necessary to retrieve feature names
        tmp_report.label = label
        tmp_report.result_path = path
        tmp_report.set_context(state.context)
        tmp_report.generate_report()

    @staticmethod
    def run_data_report(state: HPOptimizationState, report: DataReport, dataset, path: str):
        tmp_report = copy.deepcopy(report)
        tmp_report.dataset = dataset
        tmp_report.result_path = path
        tmp_report.set_context(state.context)
        tmp_report.generate_report()
