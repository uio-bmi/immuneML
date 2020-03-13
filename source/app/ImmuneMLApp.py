import os
import sys

from source.caching.CacheType import CacheType
from source.dsl.ImmuneMLParser import ImmuneMLParser
from source.dsl.semantic_model.SemanticModel import SemanticModel
from source.dsl.symbol_table.SymbolType import SymbolType
from source.environment.Constants import Constants
from source.util.PathBuilder import PathBuilder


class ImmuneMLApp:

    def __init__(self, specification_path: str, result_path: str = None):
        self._specification_path = specification_path
        self._result_path = result_path

    def set_cache(self):
        if Constants.CACHE_TYPE not in os.environ:
            os.environ[Constants.CACHE_TYPE] = CacheType.TEST.value

    def set_logging(self):
        if "ImmuneML_with_Galaxy" in os.environ and os.environ["ImmuneML_with_Galaxy"]:
            sys.stderr = open(self._result_path + "log.txt", 'w')

    def run(self):

        self.set_logging()
        self.set_cache()

        if self._result_path is not None:
            PathBuilder.build(self._result_path, warn_if_exists=True)

        symbol_table, self._specification_path = ImmuneMLParser.parse_yaml_file(self._specification_path,
                                                                                self._result_path)

        print("ImmuneML: starting the analysis...")

        instructions = symbol_table.get_by_type(SymbolType.INSTRUCTION)
        model = SemanticModel([instruction.item for instruction in instructions], self._result_path)
        model.run()


def main(argv):
    assert len(argv) == 3, "ImmuneMLApp: Some of the required parameters are missing. To run immuneML, use the command:\n" \
                               "python3 path_to_immuneMLApp.py path_to_specification.yaml analysis_result_path/"
    path = argv[2] if len(argv) == 3 else None
    app = ImmuneMLApp(argv[1], path)
    app.run()


if __name__ == "__main__":
    main(sys.argv)
