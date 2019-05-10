import sys

from source.dsl.Parser import Parser
from source.dsl.semantic_model.SemanticModel import SemanticModel


class ImmuneMLApp:

    def __init__(self, specification_path: str, result_path: str = None):
        self._specification_path = specification_path
        self._result_path = result_path

    def run(self):
        symbol_table, self._specification_path = Parser.parse_yaml_file(self._specification_path, self._result_path)

        print("ImmuneML: starting the analysis...")

        model = SemanticModel(self._result_path, self._specification_path)
        model.fill(symbol_table)
        model.execute()


if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) == 3 else None
    app = ImmuneMLApp(sys.argv[1], path)
    app.run()
