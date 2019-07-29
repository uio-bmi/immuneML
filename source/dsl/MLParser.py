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
                method, config = MLParser._parse_method(specification["ml_methods"], method_id)
                symbol_table.add(method_id, SymbolType.ML_METHOD, method, config)
        else:
            specification["ml_methods"] = {}
        return symbol_table, specification["ml_methods"]

    @staticmethod
    def _parse_method(ml_specification: dict, method_id: str) -> tuple:

        method_class = ReflectionHandler.get_class_from_path("{}/../../source/ml_methods/{}.py"
                                                             .format(EnvironmentSettings.root_path,
                                                                     ml_specification[method_id]["type"]))

        ml_specification[method_id] = {**DefaultParamsLoader.load("ml_methods/", "MLMethod"),
                                       **ml_specification[method_id]}

        return MLParser.create_method_instance(ml_specification, method_id, method_class), {
            "metrics": MLParser.map_metrics(ml_specification[method_id]),
            "model_selection_cv": ml_specification[method_id]["model_selection_cv"],
            "model_selection_n_folds": ml_specification[method_id]["model_selection_n_folds"],
            "min_example_count": ml_specification[method_id]["min_example_count"],
            "cores_for_training": ml_specification[method_id]["cores_for_training"],
            "batch_size": ml_specification[method_id]["batch_size"],
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

