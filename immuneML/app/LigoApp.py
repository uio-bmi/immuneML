import argparse
import logging
import os
import warnings
from pathlib import Path

from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.dsl.semantic_model.SemanticModel import SemanticModel
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class SimError(Exception):
    pass


class LigoApp:

    def __init__(self, specification_path: Path, result_path: Path):
        self._specification_path = Path(specification_path)
        self._result_path = Path(os.path.relpath(result_path))

        PathBuilder.build(self._result_path)

    def run(self):
        try:
            print_log(f"LIgO: parsing the specification...\n", include_datetime=True)

            symbol_table, self._specification_path = ImmuneMLParser.parse_yaml_file(self._specification_path,
                                                                                    self._result_path)

            print_log(f"LIgO: starting the simulation...\n", include_datetime=True)

            instructions = symbol_table.get_by_type(SymbolType.INSTRUCTION)
            output = symbol_table.get("output")
            model = SemanticModel([instruction.item for instruction in instructions], self._result_path, output)
            result = model.run()

            print_log(f"LIgO: finished simulation.\n", include_datetime=True)

            return result
        except SimError as e:
            print(e)


def run_ligo(namespace: argparse.Namespace):
    if os.path.isdir(namespace.result_path) and len(os.listdir(namespace.result_path)) != 0:
        raise ValueError(
            f"Directory {namespace.result_path} already exists. Please specify a new output directory for the analysis.")
    PathBuilder.build(namespace.result_path)

    logging.basicConfig(filename=Path(namespace.result_path) / "log.txt", level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(message)

    app = LigoApp(namespace.specification_path, namespace.result_path)
    app.run()


def main():
    parser = argparse.ArgumentParser(description="LIgO command line tool")
    parser.add_argument("specification_path",
                        help="Path to specification YAML file. Always used to define the simulation.")
    parser.add_argument("result_path", help="Output directory path.")

    namespace = parser.parse_args()
    namespace.specification_path = Path(namespace.specification_path)
    namespace.result_path = Path(namespace.result_path)

    run_ligo(namespace)


if __name__ == "__main__":
    main()
