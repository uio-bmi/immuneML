import copy

from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.util.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.workflows.instructions.ml_model_training.MLModelTrainingInstruction import MLModelTrainingInstruction


class MLModelTrainingParser:
    """
    Creates an instruction that will train the model on the whole dataset.

    .. highlight:: yaml
    .. code-block:: yaml

        training_ML_instruction:
            type: MLModelTraining
            dataset: dataset1
            encoding: encoding1
            preprocessing: seq1
            ml_model: ml1
            number_of_processes: 8
            metrics:
                - accuracy
                - precision
                - recall
            optimization_metric: balanced_accuracy
            labels:
                - CMV
            reports:
                data:
                    - report1
                    - report2
                encoding:
                    - report3
                models:
                    - report4

    """

    @log
    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: str = None) -> MLModelTrainingInstruction:

        self._validate_parameters(key, instruction, symbol_table)

        dataset = symbol_table.get(instruction["dataset"])
        label_config = self._create_label_config(instruction, dataset.params)
        metrics = {Metric[metric.upper()] for metric in instruction["metrics"]}
        reports = self._parse_reports(instruction, symbol_table)
        optimization_metric = Metric[instruction['optimization_metric'].upper()]
        hp_setting = self._prepare_hp_setting(symbol_table, instruction, dataset)

        ml_instruction = MLModelTrainingInstruction(dataset, metrics, optimization_metric, reports["models"], reports['encoding'], reports['data'],
                                                    instruction['number_of_processes'], label_config, hp_setting, key)

        return ml_instruction

    def _validate_parameters(self, key: str, instruction: dict, symbol_table: SymbolTable):
        location = 'MLModelTrainingParser'
        valid_metrics = [metric.name for metric in Metric]
        ParameterValidator.assert_keys(instruction.keys(), ["dataset", "type", "encoding", "preprocessing", "ml_model", "number_of_processes",
                                                            "metrics", "optimization_metric", "labels", "reports"], location, key)
        ParameterValidator.assert_all_in_valid_list([metric.upper() for metric in instruction['metrics']], valid_metrics, location, f'{key}: metrics')
        ParameterValidator.assert_in_valid_list(instruction['optimization_metric'].upper(), valid_metrics, location, f"{key}: optimization_metric")
        ParameterValidator.assert_type_and_value(instruction['labels'], list, location, 'labels')
        ParameterValidator.assert_type_and_value(instruction['dataset'], str, location, 'dataset')
        ParameterValidator.assert_in_valid_list(instruction['dataset'], symbol_table.get_keys_by_type(SymbolType.DATASET), location,
                                                f"{key}: dataset")

    def _prepare_hp_setting(self, symbol_table, instruction, dataset):
        preprocessing_sequence = symbol_table.get(instruction["preprocessing"])
        encoder = symbol_table.get(instruction["encoding"]).build_object(dataset, **symbol_table.get_config(instruction["encoding"])["encoder_params"])\
            .set_context({"dataset": dataset})

        return HPSetting(encoder, symbol_table.get_config(instruction["encoding"])["encoder_params"], symbol_table.get(instruction["ml_model"]),
                         {"model_selection_cv": symbol_table.get_config(instruction["ml_model"])["model_selection_cv"],
                          "model_selection_n_folds": symbol_table.get_config(instruction["ml_model"])["model_selection_n_folds"]},
                         preprocessing_sequence, instruction['encoding'], instruction['ml_model'], instruction['preprocessing'])

    def _prepare_context(self, instruction: dict, symbol_table: SymbolTable):
        return {"dataset": symbol_table.get(instruction["dataset"])}

    def _create_label_config(self, instruction: dict, dataset_params: dict) -> LabelConfiguration:
        labels = instruction["labels"]
        label_config = LabelConfiguration()
        for label in labels:
            label_config.add_label(label, dataset_params[label])
        return label_config

    def _parse_reports(self, instruction: dict, symbol_table: SymbolTable) -> dict:
        location = "MLModelTrainingParser"
        reports = {}

        if "reports" in instruction and instruction["reports"] is not None:

            ParameterValidator.assert_all_in_valid_list(instruction["reports"].keys(), ["data", "encoding", "models"], location, "reports")

            for report_type in instruction["reports"]:
                ParameterValidator.assert_all_in_valid_list(instruction["reports"][report_type], symbol_table.get_keys_by_type(SymbolType.REPORT),
                                                            location, f"reports: {report_type}")
                reports[report_type] = [copy.deepcopy(symbol_table.get(report_id)) for report_id in instruction["reports"][report_type]]

        reports = {**{"data": [], "encoding": [], "models": []}, **reports}

        return reports
