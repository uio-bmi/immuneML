import copy
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.Constants import Constants
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.DataWeighter import DataWeighter
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams
from immuneML.workflows.steps.DataWeighterParams import DataWeighterParams
from immuneML.workflows.steps.MLMethodAssessment import MLMethodAssessment
from immuneML.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


class HPUtil:

    @staticmethod
    def split_data(dataset: Dataset, split_config: SplitConfig, path: Path, label_config: LabelConfiguration) -> tuple:
        paths = [path / f"split_{i + 1}" for i in range(split_config.split_count)]
        params = DataSplitterParams(
            dataset=dataset,
            split_strategy=split_config.split_strategy,
            split_count=split_config.split_count,
            training_percentage=split_config.training_percentage,
            paths=paths,
            split_config=split_config,
            label_config=label_config
        )
        return DataSplitter.run(params)

    @staticmethod
    def get_average_performance(performances):
        if performances is not None and isinstance(performances, list) and len(performances) > 0 and all(isinstance(perf, float) for perf in performances):
            return sum(perf for perf in performances) / len(performances)
        else:
            return Constants.NOT_COMPUTED

    @staticmethod
    def preprocess_dataset(dataset: Dataset, preproc_sequence: list, path: Path, context: dict = None, hp_setting: HPSetting = None) -> Dataset:
        if dataset is not None:
            if isinstance(preproc_sequence, list) and len(preproc_sequence) > 0:
                PathBuilder.build(path)
                tmp_dataset = dataset.clone() if context is None or "dataset" not in context else context["dataset"]

                for preprocessing in preproc_sequence:
                    tmp_dataset = preprocessing.process_dataset(tmp_dataset, path)

                if context is not None and "dataset" in context:
                    context["preprocessed_dataset"] = {str(hp_setting): tmp_dataset}
                    indices = [i for i in range(context["dataset"].get_example_count())
                               if context["dataset"].repertoires[i].identifier in dataset.get_example_ids()]
                    preprocessed_dataset = tmp_dataset.make_subset(indices, path, Dataset.PREPROCESSED)
                else:
                    preprocessed_dataset = tmp_dataset

                return preprocessed_dataset
            else:
                return dataset

    @staticmethod
    def weight_examples(dataset, weighting_strategy: ExampleWeightingStrategy, path: Path, learn_model: bool, number_of_processes: int):
        weighted_dataset = DataWeighter.run(DataWeighterParams(
            dataset=dataset,
            weighting_strategy=weighting_strategy,
            weighting_params=ExampleWeightingParams(
                result_path=path,
                pool_size=number_of_processes,
                learn_model=learn_model
            ),
        ))
        return weighted_dataset


    @staticmethod
    def encode_dataset(dataset, hp_setting: HPSetting, path: Path, learn_model: bool, context: dict, number_of_processes: int,
                       label_configuration: LabelConfiguration, encode_labels: bool = True):
        PathBuilder.build(path)

        encoded_dataset = DataEncoder.run(DataEncoderParams(
            dataset=dataset,
            encoder=hp_setting.encoder,
            encoder_params=EncoderParams(
                model=hp_setting.encoder_params,
                result_path=path,
                pool_size=number_of_processes,
                label_config=label_configuration,
                learn_model=learn_model,
                encode_labels=encode_labels
            ),
        ))
        return encoded_dataset

    @staticmethod
    def assess_performance(method, metrics, optimization_metric, dataset, split_index, current_path: Path, test_predictions_path: Path, label: Label,
                           ml_score_path: Path):
        return MLMethodAssessment.run(MLMethodAssessmentParams(
            method=method,
            dataset=dataset,
            predictions_path=test_predictions_path,
            split_index=split_index,
            label=label,
            metrics=metrics,
            optimization_metric=optimization_metric,
            path=current_path,
            ml_score_path=ml_score_path
        ))

    @staticmethod
    def run_hyperparameter_reports(state: TrainMLModelState, path: Path) -> List[ReportResult]:
        report_results = []
        for key, report in state.reports.items():
            tmp_report = copy.deepcopy(report)
            tmp_report.state = state
            tmp_report.result_path = path / key
            tmp_report.number_of_processes = state.number_of_processes
            report_result = tmp_report.generate_report()
            report_results.append(report_result)
        return report_results

    @staticmethod
    def run_selection_reports(state: TrainMLModelState, dataset, train_datasets: list, val_datasets: list, selection_state: HPSelectionState):
        path = selection_state.path
        data_split_reports = state.selection.reports.data_split_reports.values()
        for index in range(len(train_datasets)):
            split_reports_path = path / f"split_{index + 1}"

            selection_state.train_data_reports += ReportUtil.run_data_reports(train_datasets[index], data_split_reports,
                                                                              split_reports_path / "data_reports_train",
                                                                              state.number_of_processes, state.context)
            selection_state.val_data_reports += ReportUtil.run_data_reports(val_datasets[index], data_split_reports,
                                                                            split_reports_path / "data_reports_test",
                                                                            state.number_of_processes, state.context)

        data_reports = state.selection.reports.data_reports.values()
        selection_state.data_reports = ReportUtil.run_data_reports(dataset, data_reports, path / "reports",
                                                                   state.number_of_processes, state.context)
