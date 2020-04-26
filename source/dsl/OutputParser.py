from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.util.ParameterValidator import ParameterValidator


class OutputParser:

    @staticmethod
    def parse(specs: dict, symbol_table: SymbolTable) -> dict:
        if "output" in specs:
            ParameterValidator.assert_keys(specs["output"], ["format"], "OutputParser", "output")
            ParameterValidator.assert_in_valid_list(specs["output"]["format"], ["HTML"], "OutputParser", "format")
        else:
            specs["output"] = None
        symbol_table.add("output", SymbolType.OUTPUT, specs["output"])

        return specs["output"]
