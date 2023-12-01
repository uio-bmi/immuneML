import inspect

from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class ExampleWeightingParser:
    keyword = "example_weightings"

    @staticmethod
    def parse(example_weighting_specs: dict, symbol_table: SymbolTable):
        for key in example_weighting_specs.keys():

            example_weighting, params = ExampleWeightingParser.parse_weighting_strategy(key, example_weighting_specs[key])
            symbol_table.add(key, SymbolType.WEIGHTING, example_weighting, {"example_weighting_params": params})

        return symbol_table, example_weighting_specs


    @staticmethod
    @log
    def parse_weighting_strategy(key: str, specs: dict):
        class_path = "example_weighting"

        valid_weighting_strategies = ReflectionHandler.all_nonabstract_subclasses(ExampleWeightingStrategy, subdirectory=class_path)


        weighting_strategy = ObjectParser.get_class(specs, valid_weighting_strategies, "", class_path, "ExampleWeightingParser", key)
        params = ObjectParser.get_all_params(specs, class_path, weighting_strategy.__name__, key)

        required_params = [p for p in list(inspect.signature(weighting_strategy.__init__).parameters.keys()) if p != "self"]
        ParameterValidator.assert_all_in_valid_list(params.keys(), required_params, "ExampleWeightingParser", f"{key}/{weighting_strategy.__name__}")

        return weighting_strategy, params
