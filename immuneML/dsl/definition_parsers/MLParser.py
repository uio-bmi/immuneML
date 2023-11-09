import inspect

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class MLParser:
    keyword = "ml_methods"

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

        valid_class_values = {'classifiers': ReflectionHandler.all_nonabstract_subclass_basic_names(MLMethod, "",
                                                                                                    "ml_methods/classifiers/"),
                              'gen_models': ReflectionHandler.all_nonabstract_subclass_basic_names(GenerativeModel, "",
                                                                                                   "ml_methods/generative_models/"),
                              'clustering': ReflectionHandler.all_nonabstract_subclass_basic_names(ClusteringMethod, "",
                                                                                                   "ml_methods/clustering/"),
                              'dim_reduction': ReflectionHandler.all_nonabstract_subclass_basic_names(DimRedMethod, "",
                                                                                                      "ml_methods/dim_reduction/")}

        if type(ml_specification) is str:
            ml_specification = {ml_specification: {}}

        return MLParser.parse_by_type(ml_method_id, ml_specification, valid_class_values)

    @staticmethod
    def parse_by_type(ml_method_id, ml_specification, valid_class_values):
        ml_class_name = [key for key in ml_specification.keys() if
                         key not in ['model_selection_cv', 'model_selection_n_folds']]
        assert len(ml_class_name) == 1, f"MLParser: method under {ml_method_id} is missing class name."
        ml_class_name = ml_class_name[0]

        if ml_class_name in valid_class_values['classifiers']:
            return MLParser.parse_classifiers(ml_method_id, ml_specification, ml_class_name)
        elif any(ml_class_name in class_list for class_list in [valid_class_values['gen_models'],
                                                                valid_class_values['clustering'],
                                                                valid_class_values['dim_reduction']]):
            return MLParser.parse_any_model(ml_method_id, ml_specification, ml_class_name)

        raise AssertionError(f"MLParser: {ml_class_name} is not a valid class name for any type of ML methods defined "
                             f"in immuneML. Valid values per type are: {valid_class_values}.")

    @staticmethod
    def parse_any_model(ml_method_id, ml_specification, ml_class_name):
        ml_method_class = ReflectionHandler.get_class_by_name(ml_class_name, subdirectory='ml_methods/')

        ml_specification[ml_class_name] = {
            **DefaultParamsLoader.load("ml_methods/", ml_class_name, log_if_missing=False),
            **ml_specification[ml_class_name]}

        method, params = MLParser.create_method_instance(ml_specification, ml_method_class, ml_method_id)
        ml_specification[ml_class_name] = params
        method.name = ml_method_id

        return method, ml_specification

    @staticmethod
    def parse_classifiers(ml_method_id, ml_specification, ml_class_name):
        ml_specification = {**DefaultParamsLoader.load("ml_methods/", "MLMethod"), **ml_specification}
        ml_specification_keys = list(ml_specification.keys())

        non_default_keys = [key for key in ml_specification.keys() if
                            key not in ["model_selection_cv", "model_selection_n_folds"]]

        assert len(
            ml_specification_keys) == 3, f"MLParser: ML method {ml_method_id} was not correctly specified. Expected at least 1 key " \
                                         f"(ML method name), got {len(ml_specification_keys) - 2} instead: " \
                                         f"{str([key for key in non_default_keys])[1:-1]}."

        return MLParser.parse_any_model(ml_method_id, ml_specification, ml_class_name)

    @staticmethod
    def create_method_instance(ml_specification: dict, ml_method_class, key: str) -> tuple:

        ml_params = {}

        if ml_specification[ml_method_class.__name__] is None or len(
                ml_specification[ml_method_class.__name__].keys()) == 0:
            ml_method = ml_method_class()
        else:
            ml_params = ml_specification[ml_method_class.__name__]
            init_method_keys = inspect.signature(ml_method_class.__init__).parameters.keys()
            if any([isinstance(ml_params[key], list) for key in
                    ml_params.keys()]) and "parameter_grid" in init_method_keys:

                ParameterValidator.assert_type_and_value(ml_specification['model_selection_cv'], bool,
                                                         MLParser.__name__, f'{key}: model_selection_cv')
                assert ml_specification[
                           'model_selection_cv'] is True, f"MLParser: when running ML method {key} with a list of inputs, model_selection_cv must be True! " \
                                                          f"Set the parameters for {key} to single values (not lists) or set model_selection_cv to True and model_selection_n_folds to >= 2"

                ParameterValidator.assert_type_and_value(ml_specification['model_selection_n_folds'], int,
                                                         MLParser.__name__, f'{key}: model_selection_n_folds', 2)

                ml_method = ml_method_class(
                    parameter_grid={key: [ml_params[key]] if not isinstance(ml_params[key], list) else ml_params[key]
                                    for key in ml_params.keys()})

            elif len(init_method_keys) == 3 and all(
                    arg in init_method_keys for arg in ["parameters", "parameter_grid"]):
                ml_method = ml_method_class(parameters=ml_params)
            else:
                ml_method = ml_method_class(**ml_params)

        if hasattr(ml_method, 'name'):
            ml_method.name = key

        return ml_method, ml_params
