from pathlib import Path
import pandas as pd

from immuneML.environment.SequenceType import SequenceType


class ParameterValidator:

    @staticmethod
    def assert_any_value_present(values: list, expected_values: list, location: str, parameter_name: str):
        assert any(exp_val in values for exp_val in expected_values), f"{location}: expected at least one of values {expected_values} for " \
                                                                      f"parameter {parameter_name} to be set, but none were found in {values}."

    @staticmethod
    def assert_keys_present(values: list, expected_values: list, location: str, parameter_name: str):
        for value in expected_values:
            assert value in values, f"{location}: expected {value} to be set for {parameter_name}, but got {str(values)[1:-1]} instead."

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
    def assert_type_and_value(value, parameter_type, location: str, parameter_name: str,
                              min_inclusive=None, max_inclusive=None,
                              min_exclusive=None, max_exclusive=None, exact_value=None):
        type_name = " or ".join([t.__name__ for t in parameter_type]) if type(parameter_type) is tuple else parameter_type.__name__

        base_mssg = f"{location}: {value} is not a valid value for parameter {parameter_name}. "

        assert isinstance(value, parameter_type),  f"{base_mssg}It has to be of type {type_name}, but is now of type {type(value).__name__}."

        if min_inclusive is not None:
            assert value >= min_inclusive, base_mssg + f"It has to be greater or equal to {min_inclusive}."

        if max_inclusive is not None:
            assert value <= max_inclusive, base_mssg + f"It has to be less or equal to {max_inclusive}."

        if min_exclusive is not None:
            assert value > min_exclusive, base_mssg + f"It has to be greater than {min_inclusive}."

        if max_exclusive is not None:
            assert value < max_exclusive, base_mssg + f"It has to be less than {max_inclusive}."

        if exact_value is not None:
            assert value == exact_value, base_mssg + f"It has to be equal to {exact_value}."

    @staticmethod
    def assert_keys(keys, valid_keys, location: str, parameter_name: str, exclusive: bool = True):
        for key in keys:
            assert key in valid_keys, f"{location}: {key} is not a valid parameter under {parameter_name}. " \
                                      f"Valid parameters are: {str(valid_keys)[1:-1]}."

        if exclusive:
            if len(keys) > len(valid_keys):
                raise AssertionError(f"{location}: {str(list(set(keys) - set(valid_keys)))[1:-1]} are not valid parameters "
                                     f" under {parameter_name}. Valid parameters are: {str(valid_keys)[1:-1]}. "
                                     f"Remove invalid parameters.")
            elif len(keys) < len(valid_keys):
                raise AssertionError(f"{location}: Missing parameters: {str(list(set(valid_keys) - set(keys)))[1:-1]} "
                                     f"under {parameter_name}. Valid parameters are: {str(valid_keys)[1:-1]}. "
                                     f"Please add missing parameters.")

    @staticmethod
    def assert_valid_tabular_file(file_path, location: str, parameter_name: str, sep="\t", expected_columns: list=None):
        assert Path(file_path).is_file(), f"{location}: {parameter_name} {str(file_path)} is not an existing file."

        if expected_columns is not None:
            columns = pd.read_csv(file_path, index_col=0, nrows=0, sep=sep).columns.tolist()
            assert set(columns) == set(expected_columns), f"{location}: columns for {parameter_name} are not as expected.\n" \
                                                          f"Expected: {expected_columns}\n" \
                                                          f"Found: {columns}"

    @staticmethod
    def assert_sequence_type(params, location: str = ""):
        assert "sequence_type" in params, f"{location}: 'sequence_type' is missing: {params}."
        assert params['sequence_type'].upper() in [st.name for st in
                                                   SequenceType], f"{location}: {params['sequence_type']} is not a valid sequence type. " \
                                                                  f"Valid sequence types are: {[st.name for st in SequenceType]}."
