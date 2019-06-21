from source.dsl.AssessmentType import AssessmentType
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.MetricType import MetricType
from source.ml_methods.MLMethod import MLMethod
from source.util.ReflectionHandler import ReflectionHandler


class MLParser:

    @staticmethod
    def parse(specification: dict, symbol_table: SymbolTable):

        if "ml_methods" in specification:
            for method_id in specification["ml_methods"].keys():
                parsed = MLParser._parse_method(specification["ml_methods"], method_id, symbol_table)
                symbol_table.add(method_id, SymbolType.ML_METHOD, parsed)
        else:
            specification["ml_methods"] = {}
        return symbol_table, specification["ml_methods"]

    @staticmethod
    def _parse_method(ml_specification: dict, method_id: str, symbol_table: SymbolTable) -> dict:

        assert symbol_table.contains(ml_specification[method_id]["encoding"]), \
            "MLParser: encoding {} for method {} was not properly specified, there is no such encoding defined."\
            .format(ml_specification[method_id]["encoding"], method_id)

        method_class = ReflectionHandler.get_class_from_path("{}/../../source/ml_methods/{}.py"
                                                             .format(EnvironmentSettings.root_path,
                                                                     ml_specification[method_id]["type"]))

        ml_specification[method_id] = {**DefaultParamsLoader.load("ml_methods/", "MLMethod"),
                                       **ml_specification[method_id]}

        return {
            "method": MLParser.create_method_instance(ml_specification, method_id, method_class),
            "encoding": ml_specification[method_id]["encoding"],
            "labels": ml_specification[method_id]["labels"],
            "metrics": MLParser.map_metrics(ml_specification[method_id]),
            "model_selection_cv": ml_specification[method_id]["model_selection_cv"],
            "model_selection_n_folds": ml_specification[method_id]["model_selection_n_folds"],
            "training_percentage": ml_specification[method_id]["training_percentage"],
            "split_count": ml_specification[method_id]["split_count"],
            "assessment_type": AssessmentType[ml_specification[method_id]["assessment_type"].lower()],
            "min_example_count": ml_specification[method_id]["min_example_count"],
            "cores_for_training": ml_specification[method_id]["cores_for_training"],
            "batch_size": ml_specification[method_id]["batch_size"],
            "label_to_balance": ml_specification[method_id]["label_to_balance"]
        }

    @staticmethod
    def map_metrics(method_spec: dict):
        metrics = [MetricType[metric.upper()] for metric in method_spec["metrics"]]
        return metrics

    @staticmethod
    def create_method_instance(ml_specification: dict, method_id: str, method_class) -> MLMethod:
        if "params" not in ml_specification[method_id].keys():
            method = method_class()
        elif any([isinstance(ml_specification[method_id]["params"][key], list) for key in ml_specification[method_id]["params"].keys()]):
            method = method_class(parameter_grid={key: [ml_specification[method_id]["params"][key]]
                                                  if not isinstance(ml_specification[method_id]["params"][key], list)
                                                  else ml_specification[method_id]["params"][key]
                                                  for key in ml_specification[method_id]["params"].keys()})
        else:
            method = method_class(parameters=ml_specification[method_id]["params"])
        return method

