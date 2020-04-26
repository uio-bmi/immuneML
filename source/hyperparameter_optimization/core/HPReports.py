import copy

from source.hyperparameter_optimization.states import HPItem
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from source.reports.data_reports.DataReport import DataReport
from source.reports.ml_reports.MLReport import MLReport


class HPReports:

    @staticmethod
    def run_data_reports(state: HPOptimizationState, path: str):
        if state.data_reports is not None:
            for report in state.data_reports:
                report_result = HPReports.run_data_report(state, report, state.dataset, path)
                state.data_report_results.append(report_result)

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
        for report_key, report in state.assessment.reports.data_split_reports.items():
            train_result = HPReports.run_data_report(state, report, train_val_dataset, f"{path}/reports/{report_key}/train/")
            test_result = HPReports.run_data_report(state, report, test_dataset, f"{path}/reports/{report_key}/test/")
            state.assessment_states[split_index].train_val_data_reports.append(train_result)
            state.assessment_states[split_index].test_data_reports.append(test_result)

        # optimal_model reports
        for report_key, report in state.assessment.reports.optimal_model_reports.items():
            for label in state.label_configuration.get_labels_by_name():
                opt_assesment_item = state.assessment_states[split_index].label_states[label].optimal_assessment_item
                HPReports.run_model_report(state, report, opt_assesment_item, label, f"{path}reports/{report_key}/{label}/optimal_model/")

        # model reports (all assessment models)
        for key, report in state.assessment.reports.model_reports.items():
            for label in state.label_configuration.get_labels_by_name():
                for assesment_key, assesment_item in state.assessment_states[split_index].label_states[label].assessment_items.items():
                    report_path = f"{path}reports/{key}/{label}/{assesment_key.encoder_name}/{assesment_key.ml_method_name}/"
                    if assesment_key.preproc_sequence_name is not None:
                        report_path += f"{assesment_key.preproc_sequence_name}/"

                    HPReports.run_model_report(state, report, assesment_item, label, report_path)

    @staticmethod
    def run_selection_reports(state: HPOptimizationState, dataset, train_datasets: list, val_datasets: list, selection_state: HPSelectionState):
        path = selection_state.path
        for key, report in state.selection.reports.data_split_reports.items():
            for index in range(len(train_datasets)):
                result_train = HPReports.run_data_report(state, report, train_datasets[index], path + "split_{}/reports/{}/train/".format(index + 1, key))
                result_val = HPReports.run_data_report(state, report, val_datasets[index], path + "split_{}/reports/{}/test/".format(index + 1, key))
                selection_state.train_data_reports.append(result_train)
                selection_state.val_data_reports.append(result_val)

        for key, report in state.selection.reports.data_reports.items():
            result = HPReports.run_data_report(state, report, dataset, path + "reports/{}/".format(key))
            selection_state.data_reports.append(result)

    @staticmethod
    def run_model_report(state: HPOptimizationState, report: MLReport, assessment_item: HPItem, label: str, path: str):
        tmp_report = copy.deepcopy(report)
        tmp_report.train_dataset = assessment_item.train_dataset
        tmp_report.test_dataset = assessment_item.test_dataset
        tmp_report.method = assessment_item.method
        tmp_report.ml_details_path = assessment_item.ml_details_path # Only necessary to retrieve feature names
        tmp_report.label = label
        tmp_report.result_path = path
        tmp_report.hp_setting = assessment_item.hp_setting
        tmp_report.set_context(state.context)
        report_result = tmp_report.generate_report()
        assessment_item.model_report_results.append(report_result)

    @staticmethod
    def run_data_report(state: HPOptimizationState, report: DataReport, dataset, path: str):
        tmp_report = copy.deepcopy(report)
        tmp_report.dataset = dataset
        tmp_report.result_path = path
        tmp_report.set_context(state.context)
        report_result = tmp_report.generate_report()
        return report_result
