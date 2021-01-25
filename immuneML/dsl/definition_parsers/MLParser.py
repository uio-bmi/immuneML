import inspect

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


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

        valid_class_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MLMethod, "", "ml_methods/")

        if type(ml_specification) is str:
            ml_specification = {ml_specification: {}}

        ml_specification = {**DefaultParamsLoader.load("ml_methods/", "MLMethod"), **ml_specification}
        ml_specification_keys = list(ml_specification.keys())

        ParameterValidator.assert_all_in_valid_list(list(ml_specification_keys), ["model_selection_cv", "model_selection_n_folds"] +
                                                    valid_class_values, "MLParser", ml_method_id)

        non_default_keys = [key for key in ml_specification.keys() if key not in ["model_selection_cv", "model_selection_n_folds"]]

        assert len(ml_specification_keys) == 3, f"MLParser: ML method {ml_method_id} was not correctly specified. Expected at least 1 key " \
                                                f"(ML method name), got {len(ml_specification_keys) - 2} instead: " \
                                                f"{str([key for key in non_default_keys])[1:-1]}."

        ml_method_class_name = non_default_keys[0]
        ml_method_class = ReflectionHandler.get_class_by_name(ml_method_class_name, "ml_methods/")

        ml_specification[ml_method_class_name] = {**DefaultParamsLoader.load("ml_methods/", ml_method_class_name, log_if_missing=False),
                                                  **ml_specification[ml_method_class_name]}

        method, params = MLParser.create_method_instance(ml_specification, ml_method_class, ml_method_id)
        ml_specification[ml_method_class_name] = params
        method.name = ml_method_id

        return method, ml_specification

    @staticmethod
    def create_method_instance(ml_specification: dict, ml_method_class, key: str) -> tuple:

        ml_params = {}

        if ml_specification[ml_method_class.__name__] is None or len(ml_specification[ml_method_class.__name__].keys()) == 0:
            ml_method = ml_method_class()
        else:
            ml_params = ml_specification[ml_method_class.__name__]
            init_method_keys = inspect.signature(ml_method_class.__init__).parameters.keys()
            if any([isinstance(ml_params[key], list) for key in ml_params.keys()]) and "parameter_grid" in init_method_keys:

                ParameterValidator.assert_type_and_value(ml_specification['model_selection_cv'], bool, MLParser.__name__, f'{key}: model_selection_cv', exact_value=True)
                ParameterValidator.assert_type_and_value(ml_specification['model_selection_n_folds'], int, MLParser.__name__, f'{key}: model_selection_n_folds', 2)

                ml_method = ml_method_class(parameter_grid={key: [ml_params[key]] if not isinstance(ml_params[key], list) else ml_params[key]
                                                            for key in ml_params.keys()})

            elif len(init_method_keys) == 3 and all(arg in init_method_keys for arg in ["parameters", "parameter_grid"]):
                ml_method = ml_method_class(parameters=ml_params)
            else:
                ml_method = ml_method_class(**ml_params)

        return ml_method, ml_params

