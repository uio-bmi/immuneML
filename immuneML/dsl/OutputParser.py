from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.presentation.html.HTMLBuilder import HTMLBuilder
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class OutputParser:

    @staticmethod
    def parse(specs: dict, symbol_table: SymbolTable) -> dict:
        if "output" in specs:
            ParameterValidator.assert_keys(specs["output"], ["format"], "OutputParser", "output")
            ParameterValidator.assert_in_valid_list(specs["output"]["format"], ["HTML"], "OutputParser", "format")
        else:
            specs["output"] = {"format": "HTML"}
        symbol_table.add("output", SymbolType.OUTPUT, specs["output"])

        return specs["output"]

    @staticmethod
    def generate_docs(path: Path):
        output_path = PathBuilder.build(path / "output")
        output_path = output_path / "outputs.rst"
        with output_path.open( "w") as file:
            file.writelines(HTMLBuilder.__doc__)
