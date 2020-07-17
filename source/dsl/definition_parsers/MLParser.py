import inspect

from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.logging.Logger import log
from source.ml_methods.MLMethod import MLMethod
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class MLParser:

    @staticmethod
    def parse(specification: dict, symbol_table: SymbolTable):

        for ml_method_id in specification.keys():
            ml_method, config = MLParser._parse_ml_method(ml_method_id, specification[ml_method_id])
            specification[ml_method_id] = config
            symbol_table.add(ml_method_id, SymbolType.ML_METHOD, ml_method, config)

        return symbol_table, specification

    @staticmethod
    @log
    def _parse_ml_method(ml_method_id: str, ml_specification) -> tuple:

        if type(ml_specification) is str:
            ml_specification = {ml_specification: {}}

        ml_method_class_name = [key for key in ml_specification.keys() if key not in ["model_selection_cv", "model_selection_n_folds"]][0]

        valid_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MLMethod, "", "ml_methods/")
        ParameterValidator.assert_in_valid_list(ml_method_class_name, valid_values, "MLParser", f"ML method under {ml_method_id}")

        ml_method_class = ReflectionHandler.get_class_from_path("{}/../../source/ml_methods/{}.py".format(EnvironmentSettings.root_path,
                                                                                                          ml_method_class_name))

        ml_specification = {**DefaultParamsLoader.load("ml_methods/", "MLMethod"), **ml_specification}
        ml_specification[ml_method_class_name] = {**DefaultParamsLoader.load("ml_methods/", ml_method_class_name, log_if_missing=False),
                                                  **ml_specification[ml_method_class_name]}

        method, params = MLParser.create_method_instance(ml_specification, ml_method_class)
        ml_specification[ml_method_class_name] = params

        return method, ml_specification

    @staticmethod
    def create_method_instance(ml_specification: dict, ml_method_class) -> tuple:

        ml_params = {}

        if ml_specification[ml_method_class.__name__] is None or len(ml_specification[ml_method_class.__name__].keys()) == 0:
            ml_method = ml_method_class()
        else:
            ml_params = ml_specification[ml_method_class.__name__]
            init_method_keys = inspect.signature(ml_method_class.__init__).parameters.keys()
            if any([isinstance(ml_params[key], list) for key in ml_params.keys()]) and "parameter_grid" in init_method_keys:

                ml_method = ml_method_class(parameter_grid={key: [ml_params[key]]
                                                            if not isinstance(ml_params[key], list) else ml_params[key]
                                                            for key in ml_params.keys()})
            elif len(init_method_keys) == 3 and all(arg in init_method_keys for arg in ["parameters", "parameter_grid"]):
                ml_method = ml_method_class(parameters=ml_params)
            else:
                ml_method = ml_method_class(**ml_params)

        return ml_method, ml_params

