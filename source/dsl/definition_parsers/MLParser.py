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

        classes = ReflectionHandler.get_classes_by_partial_name("", "ml_methods/")
        valid_classes = ReflectionHandler.all_nonabstract_subclasses(MLMethod)
        valid_values = [cls.__name__ for cls in valid_classes]
        ParameterValidator.assert_in_valid_list(ml_method_class_name, valid_values, "MLParser", f"ML method under {ml_method_id}")

        ml_method_class = ReflectionHandler.get_class_from_path("{}/../../source/ml_methods/{}.py".format(EnvironmentSettings.root_path,
                                                                                                          ml_method_class_name))

        ml_specification = {**DefaultParamsLoader.load("ml_methods/", "MLMethod"), **ml_specification}
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
            if any([isinstance(ml_params[key], list) for key in ml_params.keys()]):

                ml_method = ml_method_class(parameter_grid={key: [ml_params[key]]
                                                            if not isinstance(ml_params[key], list) else ml_params[key]
                                                            for key in ml_params.keys()})
            else:
                ml_method = ml_method_class(parameters=ml_params)

        return ml_method, ml_params

