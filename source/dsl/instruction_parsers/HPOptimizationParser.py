import hashlib
import os
import warnings

from source.data_model.dataset.Dataset import Dataset
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.HPOptimizationInstruction import HPOptimizationInstruction


class HPOptimizationParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable) -> HPOptimizationInstruction:

        valid_keys = ["assessment", "selection", "dataset", "strategy", "labels", "metrics", "settings", "batch_size", "type", "reports",
                      "optimization_metric"]
        ParameterValidator.assert_keys(list(instruction.keys()), valid_keys, "HPOptimizationParser", "HPOptimization")

        settings = self._parse_settings(instruction, symbol_table)
        assessment = self._parse_split_config(instruction, "assessment", symbol_table)
        selection = self._parse_split_config(instruction, "selection", symbol_table)
        dataset = symbol_table.get(instruction["dataset"])
        label_config = self._create_label_config(instruction, dataset, key)
        strategy = ReflectionHandler.get_class_by_name(instruction["strategy"], "hyperparameter_optimization/")
        metrics = {MetricType[metric.upper()] for metric in instruction["metrics"]}
        optimization_metric = MetricType[instruction["optimization_metric"].upper()]
        path = self._prepare_path(instruction)
        context = self._prepare_context(instruction, symbol_table)
        data_reports = self._prepare_reports(instruction["reports"], symbol_table)

        hp_instruction = HPOptimizationInstruction(dataset=dataset, hp_strategy=strategy(settings), hp_settings=settings,
                                                   assessment=assessment, selection=selection, metrics=metrics,
                                                   optimization_metric=optimization_metric,
                                                   label_configuration=label_config, path=path, context=context,
                                                   batch_size=instruction["batch_size"], data_reports=data_reports, name=key)

        return hp_instruction

    def _prepare_reports(self, reports: list, symbol_table: SymbolTable) -> dict:
        if reports is not None:
            return {report_id: symbol_table.get(report_id) for report_id in reports}
        else:
            return {}

    def _prepare_context(self, instruction: dict, symbol_table: SymbolTable):
        return {"dataset": symbol_table.get(instruction["dataset"])}

    def _parse_settings(self, instruction: dict, symbol_table: SymbolTable) -> list:
        try:
            settings = []
            for setting in instruction["settings"]:
                if "preprocessing" in setting and symbol_table.contains(setting["preprocessing"]):
                    preprocessing_sequence = symbol_table.get(setting["preprocessing"])
                    preproc_name = setting["preprocessing"]
                else:
                    preprocessing_sequence = []
                    preproc_name = None

                s = HPSetting(encoder=symbol_table.get(setting["encoding"]), encoder_name=setting["encoding"],
                              encoder_params=symbol_table.get_config(setting["encoding"])["encoder_params"],
                              ml_method=symbol_table.get(setting["ml_method"]), ml_method_name=setting["ml_method"],
                              ml_params=symbol_table.get_config(setting["ml_method"]),
                              preproc_sequence=preprocessing_sequence, preproc_sequence_name=preproc_name)
                settings.append(s)
            return settings
        except KeyError as key_error:
            print(f"HPOptimizationParser: parameter {key_error.args[0]} was not defined under settings in HPOptimization instruction.")
            raise key_error

    def _prepare_path(self, instruction: dict) -> str:
        if "path" in instruction:
            path = os.path.abspath(instruction["path"])
        else:
            path = EnvironmentSettings.default_analysis_path + hashlib.md5(str(instruction).encode()).hexdigest()

        return path

    def _create_label_config(self, instruction: dict, dataset: Dataset, instruction_key: str) -> LabelConfiguration:
        labels = instruction["labels"]
        label_config = LabelConfiguration()
        for label in labels:
            if dataset.params is not None and label in dataset.params:
                label_values = dataset.params[label]
            elif hasattr(dataset, "get_metadata"):
                label_values = list(set(dataset.get_metadata([label])[label]))
            else:
                label_values = []
                warnings.warn(f"HPOptimizationParser: for instruction {instruction_key}, label values could not be recovered for label "
                              f"{label}, using empty list instead.  This could cause problems with some encodings. "
                              f"If that might be the case, check if the dataset {dataset.name} has been properly loaded.")

            label_config.add_label(label, label_values)
        return label_config

    def _parse_split_config(self, instruction: dict, key: str, symbol_table: SymbolTable) -> SplitConfig:

        try:

            default_params = DefaultParamsLoader.load("instructions/", SplitConfig.__name__)

            if "reports" in instruction[key]:
                report_config_input = {report_type: {report_id: symbol_table.get(report_id) for report_id in instruction[key]["reports"][report_type]}
                                       for report_type in instruction[key]["reports"]}
            else:
                report_config_input = {}

            instruction[key] = {**default_params, **instruction[key]}

            return SplitConfig(split_strategy=SplitType[instruction[key]["split_strategy"].upper()],
                               split_count=int(instruction[key]["split_count"]),
                               training_percentage=float(instruction[key]["training_percentage"]),
                               reports=ReportConfig(**report_config_input))

        except KeyError as key_error:
            print(f"HPOptimizationParser: parameter {key_error.args[0]} was not defined under {key}.")
            raise key_error
