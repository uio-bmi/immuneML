import sys

from source.dsl.ImmuneMLParser import ImmuneMLParser
from source.dsl.SymbolType import SymbolType
from source.dsl.semantic_model.SemanticModel import SemanticModel


class ImmuneMLApp:

    def __init__(self, specification_path: str, result_path: str = None):
        self._specification_path = specification_path
        self._result_path = result_path

    def run(self):
        symbol_table, self._specification_path = ImmuneMLParser.parse_yaml_file(self._specification_path,
                                                                                self._result_path)

        print("ImmuneML: starting the analysis...")

        instructions = symbol_table.get_by_type(SymbolType.INSTRUCTION)
        model = SemanticModel([instruction[1] for instruction in instructions], self._result_path)
        model.run()


if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) == 3 else None
    app = ImmuneMLApp(sys.argv[1], path)
    app.run()
