from source.dsl.AssessmentType import AssessmentType
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
        return symbol_table, {}

    @staticmethod
    def _parse_method(ml_specification: dict, method_id: str, symbol_table: SymbolTable) -> dict:

        assert symbol_table.contains(ml_specification[method_id]["encoding"]), \
            "MLParser: encoding {} for method {} was not properly specified, there is no such encoding defined."\
            .format(ml_specification[method_id]["encoding"], method_id)

        method_class = ReflectionHandler.get_class_from_path("{}/../../source/ml_methods/{}.py"
                                                             .format(EnvironmentSettings.root_path,
                                                                     ml_specification[method_id]["type"]))

        method_dict = ml_specification[method_id]

        return {
            "method": MLParser.create_method_instance(ml_specification, method_id, method_class),
            "encoding": method_dict["encoding"],
            "labels": method_dict["labels"],
            "metrics": MLParser.map_metrics(ml_specification[method_id]),
            "model_selection_cv": method_dict["model_selection_cv"],
            "model_selection_n_folds": method_dict["model_selection_n_folds"],
            "split_count": method_dict["split_count"],
            "assessment_type": AssessmentType[method_dict["assessment_type"].lower()]
        }

    @staticmethod
    def map_metrics(method_spec: dict):
        metrics = [MetricType[metric.upper()] for metric in method_spec["metrics"]]
        return metrics

    @staticmethod
    def create_method_instance(ml_specification: dict, method_id: str, method_class) -> MLMethod:
        if any([isinstance(ml_specification[method_id]["params"][key], list) for key in ml_specification[method_id]["params"].keys()]):
            method = method_class(parameter_grid={key: [ml_specification[method_id]["params"][key]]
                                                  if not isinstance(ml_specification[method_id]["params"][key], list)
                                                  else ml_specification[method_id]["params"][key]
                                                  for key in ml_specification[method_id]["params"].keys()})
        else:
            method = method_class(parameters=ml_specification[method_id]["params"])
        return method

