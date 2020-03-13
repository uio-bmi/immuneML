import inspect

from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class ObjectParser:

    @staticmethod
    def get_class_name(specs, valid_class_names, class_name_ending, location, key):
        if isinstance(specs, str):
            class_name = specs
        elif isinstance(specs, dict):
            assert len(specs) == 1, \
                f"{location}: More than one parameter passed to {class_name_ending.lower()} under key {key}. " \
                f"Only one value can be specified here. Valid options are: {str(valid_class_names)[1:-1]}"
            class_name = list(specs.keys())[0]
        else:
            raise AssertionError(f"{location}: Incorrect specification under key {key}. Correct specification would be: "
                                 f"\n{key}: {valid_class_names[0]}\n"
                                 f"For more information and details on how to specify parameters, please refer to the documentation.")
        return class_name

    @staticmethod
    def get_params(specs, class_name):
        params = {}
        if isinstance(specs, dict):
            params = specs[class_name]
        return params

    @staticmethod
    def get_class(specs, valid_class_names, class_name_ending, class_path, location, key):
        class_name = ObjectParser.get_class_name(specs, valid_class_names, class_name_ending, location, key)
        cls = ReflectionHandler.get_class_by_name(f"{class_name}{class_name_ending}", class_path)
        return cls

    @staticmethod
    def get_all_params(specs, class_path, short_class_name):
        default_params = DefaultParamsLoader.load(class_path, short_class_name)
        specified_params = ObjectParser.get_params(specs, short_class_name)
        params = {**default_params, **specified_params}
        return params

    @staticmethod
    def parse_object(specs, valid_class_names: list, class_name_ending: str, class_path: str, location: str, key: str,
                     builder: bool = False, return_params_dict: bool = False):

        class_name = ObjectParser.get_class_name(specs, valid_class_names, class_name_ending, location, key)
        ParameterValidator.assert_in_valid_list(class_name, valid_class_names, location, key)

        cls = ReflectionHandler.get_class_by_name(f"{class_name}{class_name_ending}", class_path)
        params = ObjectParser.get_all_params(specs, class_path, class_name)

        try:
            obj = cls.build_object(**params) if builder else cls(**params)
        except TypeError as err:
            raise AssertionError(f"{location}: invalid parameter {err.args[0]} when specifying parameters in {specs} "
                                 f"under key {key}. Valid parameter names are: "
                                 f"{[name for name in inspect.signature(cls.__init__).keys()]}")

        return (obj, params) if return_params_dict else obj
