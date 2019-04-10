import sys

from source.dsl.Parser import Parser
from source.dsl.semantic_model.SemanticModel import SemanticModel


class ImmuneMLApp:

    def __init__(self, specification_path: str, result_path: str = None):
        self._specification_path = specification_path
        self._result_path = result_path

    def run(self):
        symbol_table = Parser.parse_yaml_file(self._specification_path)

        print("ImmuneML: starting the analysis...")

        model = SemanticModel(self._result_path)
        model.fill(symbol_table)
        model.execute()


if __name__ == "__main__":
    app = ImmuneMLApp(sys.argv[1])
    app.run()
