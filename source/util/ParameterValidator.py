class ParameterValidator:

    @staticmethod
    def assert_all_in_valid_list(values: list, valid_values: list, location: str, parameter_name: str):
        for value in values:
            ParameterValidator.assert_in_valid_list(value, valid_values, location, parameter_name)

    @staticmethod
    def assert_in_valid_list(value, valid_values: list, location: str, parameter_name: str):
        assert value in valid_values, \
            f"{location}: {value} is not a valid value for parameter {parameter_name}. Valid values are {str(valid_values)[1:-1]}."

    @staticmethod
    def assert_all_type_and_value(values, parameter_type, location: str, parameter_name: str, min_inclusive=None, max_inclusive=None):
        for value in values:
            ParameterValidator.assert_type_and_value(value, parameter_type, location, parameter_name, min_inclusive, max_inclusive)

    @staticmethod
    def assert_type_and_value(value, parameter_type, location: str, parameter_name: str, min_inclusive=None, max_inclusive=None):
        assert isinstance(value, parameter_type), f"{location}: {value} is not a valid value for parameter {parameter_name}. " \
                                                  f"It has to be of type {parameter_type}, but is now of type {type(value)}."

        if min_inclusive:
            assert value >= min_inclusive, f"{location}: {value} is not a valid value for parameter {parameter_name}. " \
                                           f"It has to be greater or equal to {min_inclusive}."

        if max_inclusive:
            assert value <= min_inclusive, f"{location}: {value} is not a valid value for parameter {parameter_name}. " \
                                           f"It has to be less or equal to {max_inclusive}."

    @staticmethod
    def assert_keys(keys, valid_keys, location: str, parameter_name: str, exclusive: bool = True):
        for key in keys:
            assert key in valid_keys, f"{location}: {key} is not a valid parameter name under {parameter_name}. " \
                                      f"Valid parameter names are: {str(valid_keys)[1:-1]}."

        if exclusive:
            if len(keys) > len(valid_keys):
                raise AssertionError(f"{location}: {str(list(set(keys) - set(valid_keys)))[1:-1]} are not valid parameter "
                                     f"names under {parameter_name}. Valid parameter names are: {str(valid_keys)[1:-1]}. "
                                     f"Remove invalid names.")
            elif len(keys) < len(valid_keys):
                raise AssertionError(f"{location}: Missing names: {str(list(set(valid_keys) - set(keys)))[1:-1]} "
                                     f"under {parameter_name}. Valid parameter names are: {str(valid_keys)[1:-1]}. "
                                     f"Add missing names.")
