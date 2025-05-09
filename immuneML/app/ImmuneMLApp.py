import argparse
import logging
import os
import shutil
import traceback
from datetime import datetime
from sys import exit as sys_exit
import warnings
from pathlib import Path

from immuneML.caching.CacheType import CacheType
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.dsl.semantic_model.SemanticModel import SemanticModel
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


class ImmuneMLApp:

    def __init__(self, specification_path: Path, result_path: Path, logging_level: str = 'INFO'):

        self._specification_path = Path(specification_path)
        self._result_path = Path(os.path.relpath(result_path))

        PathBuilder.build(self._result_path)

        logging.basicConfig(filename=Path(self._result_path) / "log.txt", level=getattr(logging, logging_level.upper()),
                            format='%(asctime)s %(levelname)s: %(message)s', force=True)

        self._cache_path = self._result_path / "cache"

    def set_cache(self):
        os.environ[Constants.CACHE_TYPE] = CacheType.PRODUCTION.value
        EnvironmentSettings.set_cache_path(self._cache_path)

    def clear_cache(self):
        shutil.rmtree(self._cache_path, ignore_errors=True)
        EnvironmentSettings.reset_cache_path()
        del os.environ[Constants.CACHE_TYPE]

    def run(self):
        try:
            print_log(f"Running immuneML version {Constants.VERSION}\n", include_datetime=True)

            self.set_cache()

            print_log(f"immuneML: parsing the specification...\n", include_datetime=True)

            symbol_table, self._specification_path = ImmuneMLParser.parse_yaml_file(self._specification_path,
                                                                                    self._result_path)

            print_log(f"immuneML: starting the analysis...\n", include_datetime=True)

            instructions = symbol_table.get_by_type(SymbolType.INSTRUCTION)
            output = symbol_table.get("output")
            model = SemanticModel([instruction.item for instruction in instructions], self._result_path, output)
            result = model.run()

            self.clear_cache()

            print_log(f"ImmuneML: finished analysis.\n", include_datetime=True)

            return result

        except (ModuleNotFoundError, ImportError) as e:
            sys_exit(f"{e}\n\nAn error occurred when trying to import a package. Please check if all necessary "
                     f"packages are installed correctly. See the log above for more details.")
        except Exception as e:
            traceback.print_exc()
            raise e


def run_immuneML(namespace: argparse.Namespace):
    if os.path.isdir(namespace.result_path) and len(os.listdir(namespace.result_path)) != 0:
        result_path = f"{namespace.result_path}_{datetime.now()}"
        print(f"Directory {namespace.result_path} already exists. The output of this analysis will be "
              f"stored in {result_path}.")
    else:
        result_path = namespace.result_path

    PathBuilder.build(result_path)

    if namespace.tool is None:
        app = ImmuneMLApp(namespace.specification_path, result_path, namespace.logging)
    else:
        app_cls = ReflectionHandler.get_class_by_name(namespace.tool, "api/")
        app = app_cls(**vars(namespace))

    app.run()


def main():
    parser = argparse.ArgumentParser(description="immuneML command line tool")
    parser.add_argument("specification_path", help="Path to specification YAML file. Always used to define the "
                                                   "analysis.")
    parser.add_argument("result_path", help="Output directory path.")
    parser.add_argument("--tool", help="Name of the tool which calls immuneML. This name will be used to invoke "
                                       "appropriate API call, which will then do additional work in tool-dependent "
                                       "way before running standard immuneML.")
    parser.add_argument('--logging', help='Logging level to use',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument("--version", action="version", version=Constants.VERSION)

    namespace = parser.parse_args()
    namespace.specification_path = Path(namespace.specification_path)
    namespace.result_path = Path(namespace.result_path)

    run_immuneML(namespace)


if __name__ == "__main__":
    main()
