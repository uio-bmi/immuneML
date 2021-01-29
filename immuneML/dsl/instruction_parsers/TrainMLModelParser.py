import hashlib
import warnings
from inspect import signature
from pathlib import Path
from typing import Tuple

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.definition_parsers.PreprocessingParser import PreprocessingParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Metric import Metric
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.LeaveOneOutConfig import LeaveOneOutConfig
from immuneML.hyperparameter_optimization.config.ManualSplitConfig import ManualSplitConfig
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TrainMLModelParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> TrainMLModelInstruction:

        valid_keys = ["assessment", "selection", "dataset", "strategy", "labels", "metrics", "settings", "number_of_processes", "type", "reports",
                      "optimization_metric", 'refit_optimal_model', 'store_encoded_data']
        ParameterValidator.assert_type_and_value(instruction['settings'], list, TrainMLModelParser.__name__, 'settings')
        ParameterValidator.assert_keys(list(instruction.keys()), valid_keys, TrainMLModelParser.__name__, "TrainMLModel")
        ParameterValidator.assert_type_and_value(instruction['refit_optimal_model'], bool, TrainMLModelParser.__name__, 'refit_optimal_model')
        ParameterValidator.assert_type_and_value(instruction['metrics'], list, TrainMLModelParser.__name__, 'metrics')
        ParameterValidator.assert_type_and_value(instruction['optimization_metric'], str, TrainMLModelParser.__name__, 'optimization_metric')
        ParameterValidator.assert_type_and_value(instruction['number_of_processes'], int, TrainMLModelParser.__name__, 'number_of_processes')
        ParameterValidator.assert_type_and_value(instruction['strategy'], str, TrainMLModelParser.__name__, 'strategy')
        ParameterValidator.assert_type_and_value(instruction['store_encoded_data'], bool, TrainMLModelParser.__name__, 'store_encoded_data')

        settings = self._parse_settings(instruction, symbol_table)
        dataset = symbol_table.get(instruction["dataset"])
        assessment = self._parse_split_config(key, instruction, "assessment", symbol_table, len(settings))
        selection = self._parse_split_config(key, instruction, "selection", symbol_table, len(settings))
        assessment, selection = self._update_split_configs(assessment, selection, dataset)
        label_config = self._create_label_config(instruction, dataset, key)
        strategy = ReflectionHandler.get_class_by_name(instruction["strategy"], "hyperparameter_optimization/")
        metrics = {Metric[metric.upper()] for metric in instruction["metrics"]}
        optimization_metric = Metric[instruction["optimization_metric"].upper()]
        metric_search_criterion = Metric.get_search_criterion(optimization_metric)
        path = self._prepare_path(instruction)
        context = self._prepare_context(instruction, symbol_table)
        reports = self._prepare_reports(instruction["reports"], symbol_table)

        hp_instruction = TrainMLModelInstruction(dataset=dataset, hp_strategy=strategy(settings, metric_search_criterion),
                                                 hp_settings=settings, assessment=assessment, selection=selection, metrics=metrics,
                                                 optimization_metric=optimization_metric, refit_optimal_model=instruction['refit_optimal_model'],
                                                 label_configuration=label_config, path=path, context=context,
                                                 store_encoded_data=instruction['store_encoded_data'],
                                                 number_of_processes=instruction["number_of_processes"], reports=reports, name=key)

        return hp_instruction

    def _update_split_configs(self, assessment: SplitConfig, selection: SplitConfig, dataset: Dataset) -> Tuple[SplitConfig, SplitConfig]:

        if assessment.split_strategy == SplitType.LOOCV:
            assessment.split_count = dataset.get_example_count()
            train_val_example_count = assessment.split_count - 1
        elif assessment.split_strategy == SplitType.K_FOLD:
            train_val_example_count = int(dataset.get_example_count() * (assessment.split_count - 1) / assessment.split_count)
        else:
            train_val_example_count = int(dataset.get_example_count() * assessment.training_percentage)

        if selection.split_strategy == SplitType.LOOCV:
            selection.split_count = train_val_example_count

        return assessment, selection

    def _prepare_reports(self, reports: list, symbol_table: SymbolTable) -> dict:
        if reports is not None:
            report_objects = {report_id: symbol_table.get(report_id) for report_id in reports}
            ParameterValidator.assert_all_type_and_value(report_objects.values(), TrainMLModelReport, TrainMLModelParser.__name__, 'reports')
            return report_objects
        else:
            return {}

    def _prepare_context(self, instruction: dict, symbol_table: SymbolTable):
        return {"dataset": symbol_table.get(instruction["dataset"])}

    def _parse_settings(self, instruction: dict, symbol_table: SymbolTable) -> list:
        try:
            settings = []
            for index, setting in enumerate(instruction["settings"]):
                if "preprocessing" in setting:
                    ParameterValidator.assert_type_and_value(setting["preprocessing"], str, TrainMLModelParser.__name__, f'settings: {index+1}. '
                                                                                                                         f'element: preprocessing')
                    if symbol_table.contains(setting["preprocessing"]):
                        preprocessing_sequence = symbol_table.get(setting["preprocessing"])
                        preproc_name = setting["preprocessing"]
                    else:
                        raise KeyError(f"{TrainMLModelParser.__name__}: preprocessing was set in the TrainMLModel instruction to value "
                                       f"{setting['preprocessing']}, but no such preprocessing was defined in the specification under "
                                       f"definitions: {PreprocessingParser.keyword}.")
                else:
                    setting["preprocessing"] = None
                    preprocessing_sequence = []
                    preproc_name = None

                ParameterValidator.assert_keys(setting.keys(), ["preprocessing", "ml_method", "encoding"], TrainMLModelParser.__name__,
                                               f"settings, {index + 1}. entry")

                encoder = symbol_table.get(setting["encoding"]).build_object(symbol_table.get(instruction["dataset"]),
                                                                             **symbol_table.get_config(setting["encoding"])["encoder_params"])\
                    .set_context({"dataset": symbol_table.get(instruction['dataset'])})

                s = HPSetting(encoder=encoder,
                              encoder_name=setting["encoding"],
                              encoder_params=symbol_table.get_config(setting["encoding"])["encoder_params"],
                              ml_method=symbol_table.get(setting["ml_method"]), ml_method_name=setting["ml_method"],
                              ml_params=symbol_table.get_config(setting["ml_method"]),
                              preproc_sequence=preprocessing_sequence, preproc_sequence_name=preproc_name)
                settings.append(s)
            return settings
        except KeyError as key_error:
            raise KeyError(f"{TrainMLModelParser.__name__}: parameter {key_error.args[0]} was not defined under settings in TrainMLModel instruction.")

    def _prepare_path(self, instruction: dict) -> Path:
        if "path" in instruction:
            path = Path(instruction["path"]).absolute()
        else:
            path = EnvironmentSettings.default_analysis_path / hashlib.md5(str(instruction).encode()).hexdigest()

        return path

    def _check_label_format(self, labels: list, instruction_key: str):
        ParameterValidator.assert_type_and_value(labels, list, TrainMLModelParser.__name__, f'{instruction_key}/labels')
        assert all(isinstance(label, str) or isinstance(label, dict) for label in labels), \
            f"{TrainMLModelParser.__name__}: labels under {instruction_key} were not defined properly. The list of labels has to either be a list of " \
            f"label names, or there can be a parameter 'positive_class' defined under the label name."

        assert all(len(list(label.keys())) == 1 and isinstance(list(label.values())[0], dict) and 'positive_class' in list(label.values())[0]
                   and len(list(list(label.values())[0].keys())) == 1 for label in [l for l in labels if isinstance(l, dict)]), \
            f"{TrainMLModelParser.__name__}: labels that are specified by more than label name, can include only one parameter called 'positive_class'."

    def _create_label_config(self, instruction: dict, dataset: Dataset, instruction_key: str) -> LabelConfiguration:
        labels = instruction["labels"]

        self._check_label_format(labels, instruction_key)

        label_config = LabelConfiguration()
        for label in labels:
            label_name = label if isinstance(label, str) else list(label.keys())[0]
            positive_class = label[label_name]['positive_class'] if isinstance(label, dict) else None
            if dataset.labels is not None and label_name in dataset.labels:
                label_values = dataset.labels[label_name]
            elif hasattr(dataset, "get_metadata"):
                label_values = list(set(dataset.get_metadata([label_name])[label_name]))
            else:
                label_values = []
                warnings.warn(f"{TrainMLModelParser.__name__}: for instruction {instruction_key}, label values could not be recovered for label "
                              f"{label}, using empty list instead.  This could cause problems with some encodings. "
                              f"If that might be the case, check if the dataset {dataset.name} has been properly loaded.")

            label_config.add_label(label_name, label_values, positive_class=positive_class)
        return label_config

    def _parse_split_config(self, instruction_key, instruction: dict, split_key: str, symbol_table: SymbolTable, settings_count: int) -> SplitConfig:

        try:

            default_params = DefaultParamsLoader.load("instructions/", SplitConfig.__name__)
            report_config_input = self._prepare_report_config(instruction_key, instruction, split_key, symbol_table)
            instruction[split_key] = {**default_params, **instruction[split_key]}

            split_strategy = SplitType[instruction[split_key]["split_strategy"].upper()]
            training_percentage = float(instruction[split_key]["training_percentage"]) if split_strategy == SplitType.RANDOM else -1

            if split_strategy == SplitType.RANDOM and training_percentage == 1 and settings_count > 1:
                raise ValueError(f"{TrainMLModelParser.__name__}: all data under {instruction_key}/{split_key} was specified to be used for "
                                 f"training, but {settings_count} settings were specified for evaluation. Please define a test/validation set by "
                                 f"reducing the training percentage (e.g., to 0.7) or use only one hyperparameter setting to run the analysis.")

            return SplitConfig(split_strategy=split_strategy,
                               split_count=int(instruction[split_key]["split_count"]),
                               training_percentage=training_percentage,
                               reports=ReportConfig(**report_config_input),
                               manual_config=ManualSplitConfig(**instruction[split_key]["manual_config"]) if "manual_config" in instruction[split_key] else None,
                               leave_one_out_config=LeaveOneOutConfig(**instruction[split_key]["leave_one_out_config"])
                               if "leave_one_out_config" in instruction[split_key] else None)

        except KeyError as key_error:
            raise KeyError(f"{TrainMLModelParser.__name__}: parameter {key_error.args[0]} was not defined under {split_key}.")

    def _prepare_report_config(self, instruction_key, instruction, split_key, symbol_table):
        if "reports" in instruction[split_key]:
            location = f"{instruction_key}/{split_key}/reports"
            report_types = list(signature(ReportConfig).parameters.keys())
            ParameterValidator.assert_all_in_valid_list(instruction[split_key]["reports"].keys(), report_types,
                                                        location, "reports")

            for report_type in instruction[split_key]["reports"]:
                ParameterValidator.assert_type_and_value(instruction[split_key]["reports"][report_type], list, f"{location}/{report_type}",
                                                         report_type)

            report_config_input = {report_type: {report_id: symbol_table.get(report_id) for report_id in instruction[split_key]["reports"][report_type]}
                                   for report_type in instruction[split_key]["reports"]}
        else:
            report_config_input = {}

        return report_config_input
