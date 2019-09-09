import copy

from source.dsl.SymbolTable import SymbolTable
from source.util.ReflectionHandler import ReflectionHandler


class ParameterParser:
    """
    ParameterParser calls parse() method of a given parameter parser class if it exists
    and if it does not, then returns a tuple consisting of two dictionaries with the same content:
    parsed parameters and parameter specification.

    This ensures it is possible to add the encoding / report / preprocessing without having to write parsers for the DSL:
    the parameter resolution (e.g. loading sequences from a given file path) will then be the responsibility
    of the new encoding / report / preprocessing class.
    """

    @staticmethod
    def parse(params, class_name: str = "", subdirectory: str = "", symbol_table: SymbolTable = None):
        if class_name != "" and ReflectionHandler.exists("{}Parser".format(class_name), subdirectory):
            parser_class = ReflectionHandler.get_class_by_name("{}Parser".format(class_name))
            parsed_params, params_specs = parser_class.parse(params)
        else:
            parsed_params = copy.deepcopy(params)
            params_specs = parsed_params
        return parsed_params, params_specs
