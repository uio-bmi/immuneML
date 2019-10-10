import hashlib
import os

from source.dsl.SymbolTable import SymbolTable
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.ReportConfig import ReportConfig
from source.hyperparameter_optimization.SplitConfig import SplitConfig
from source.hyperparameter_optimization.SplitType import SplitType
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.processes.HPOptimizationProcess import HPOptimizationProcess


class HPOptimizationParser:

    def parse(self, instruction: dict, symbol_table: SymbolTable) -> HPOptimizationProcess:
        settings = self._parse_settings(instruction, symbol_table)
        assessment = self._parse_split_config(instruction, "assessment", symbol_table)
        selection = self._parse_split_config(instruction, "selection", symbol_table)
        dataset = symbol_table.get(instruction["dataset"])
        label_config = self._create_label_config(instruction, dataset.params)
        strategy = ReflectionHandler.get_class_by_name(instruction["strategy"], "hyperparameter_optimization/")
        metrics = {MetricType[metric.upper()] for metric in instruction["metrics"]}
        path = self._prepare_path(instruction)
        context = self._prepare_context(instruction, symbol_table)

        hp_process = HPOptimizationProcess(dataset=dataset, hp_strategy=strategy(settings), hp_settings=settings,
                                           assessment=assessment, selection=selection, metrics=metrics,
                                           label_configuration=label_config, path=path, context=context)

        return hp_process

    def _prepare_context(self, instruction: dict, symbol_table: SymbolTable):
        return {"dataset": symbol_table.get(instruction["dataset"])}

    def _parse_settings(self, instruction: dict, symbol_table: SymbolTable) -> list:
        settings = []
        for setting in instruction["settings"]:
            if "preprocessing" in setting and symbol_table.contains(setting["preprocessing"]):
                preprocessing_sequence = symbol_table.get(setting["preprocessing"])
            else:
                preprocessing_sequence = []

            s = HPSetting(encoder=symbol_table.get(setting["encoding"]),
                          encoder_params=symbol_table.get_config(setting["encoding"])["encoder_params"],
                          ml_method=symbol_table.get(setting["ml_method"]),
                          ml_params={"model_selection_cv": symbol_table.get_config(setting["ml_method"])["model_selection_cv"],
                                     "model_selection_n_folds":
                                         symbol_table.get_config(setting["ml_method"])["model_selection_n_folds"]},
                          preproc_sequence=preprocessing_sequence)
            settings.append(s)
        return settings

    def _prepare_path(self, instruction: dict) -> str:
        if "path" in instruction:
            path = os.path.abspath(instruction["path"])
        else:
            path = EnvironmentSettings.default_analysis_path + hashlib.md5(str(instruction).encode()).hexdigest()

        return path

    def _create_label_config(self, instruction: dict, dataset_params: dict) -> LabelConfiguration:
        labels = instruction["labels"]
        label_config = LabelConfiguration()
        for label in labels:
            label_config.add_label(label, dataset_params[label])
        return label_config

    def _parse_split_config(self, instruction: dict, key: str, symbol_table: SymbolTable) -> SplitConfig:

        if "reports" in instruction[key]:
            report_config_input = {report_type: [symbol_table.get(report_id) for report_id in instruction[key]["reports"][report_type]]
                                   for report_type in instruction[key]["reports"]}
        else:
            report_config_input = {}

        return SplitConfig(split_strategy=SplitType[instruction[key]["split_strategy"].upper()],
                           split_count=int(instruction[key]["split_count"]),
                           training_percentage=float(instruction[key]["training_percentage"]),
                           label_to_balance=instruction[key]["label_to_balance"],
                           reports=ReportConfig(**report_config_input))
