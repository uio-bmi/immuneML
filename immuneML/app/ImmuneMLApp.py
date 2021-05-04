import argparse
import datetime
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

from immuneML.caching.CacheType import CacheType
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.dsl.semantic_model.SemanticModel import SemanticModel
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.info import PROGRAM_NAME, URL, CITE, LOGO, VERSION


def hello_world(infile, rundir, outpath):
    """Print out a polite greeting for immuneML.

    Parameters
    ----------
    infile : string
        String showing the location of the input file.
    rundir : string
        String showing the location we are running in.
    outpath : string
        The output path.
    """
    timestart = datetime.datetime.now()
    pyversion = sys.version.split()[0]
    print('\n'.join([LOGO]))
    print(f'{PROGRAM_NAME} version: {VERSION}')
    print(f'Start of execution: {timestart}')
    print(f'Python version: {pyversion}')
    print(f'Running in directory: {rundir}')
    print(f'Specification file: {infile}')
    print(f'Output path: {outpath}')


def bye_bye_world():
    """Print out the goodbye message for immuneML."""
    timeend = datetime.datetime.now()
    print()
    print(f'End of {PROGRAM_NAME}, execution: {timeend}')
    # display some references:
    references = ['{} references:'.format(PROGRAM_NAME)]
    references.append(('-')*len(references[0]))
    for line in CITE.split('\n'):
        if line:
            references.append(line)
    print('\n'.join(references))
    print(f'{URL}')


class ImmuneMLApp:

    def __init__(self, specification_path: Path, result_path: Path):
        self._specification_path = Path(specification_path)
        self._result_path = Path(os.path.relpath(result_path))

        PathBuilder.build(self._result_path)

        self._cache_path = self._result_path / "cache"

    def set_cache(self):
        os.environ[Constants.CACHE_TYPE] = CacheType.PRODUCTION.value
        EnvironmentSettings.set_cache_path(self._cache_path)

    def clear_cache(self):
        shutil.rmtree(self._cache_path, ignore_errors=True)
        EnvironmentSettings.reset_cache_path()
        del os.environ[Constants.CACHE_TYPE]

    def run(self):

        self.set_cache()

        print(f"{datetime.datetime.now()}: ImmuneML: parsing the specification...\n", flush=True)

        symbol_table, self._specification_path = ImmuneMLParser.parse_yaml_file(self._specification_path, self._result_path)

        print(f"{datetime.datetime.now()}: ImmuneML: starting the analysis...\n", flush=True)

        instructions = symbol_table.get_by_type(SymbolType.INSTRUCTION)
        output = symbol_table.get("output")
        model = SemanticModel([instruction.item for instruction in instructions], self._result_path, output)
        result = model.run()

        self.clear_cache()

        print(f"{datetime.datetime.now()}: ImmuneML: finished analysis.\n", flush=True)

        return result


def run_immuneML(namespace: argparse.Namespace):
    if os.path.isdir(namespace.result_path) and len(os.listdir(namespace.result_path)) != 0:
        raise ValueError(f"Directory {namespace.result_path} already exists. Please specify a new output directory for the analysis.")
    PathBuilder.build(namespace.result_path)

    logging.basicConfig(filename=Path(namespace.result_path) / "log.txt", level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(message)

    if namespace.tool is None:
        app = ImmuneMLApp(namespace.specification_path, namespace.result_path)
    else:
        app_cls = ReflectionHandler.get_class_by_name(namespace.tool, "api/")
        app = app_cls(**vars(namespace))

    app.run()


def main():
    parser = argparse.ArgumentParser(description=f"{PROGRAM_NAME} command line tool")
    parser.add_argument("specification_path", help="Path to specification YAML file. Always used to define the analysis.")
    parser.add_argument("result_path", help="Output directory path.")
    parser.add_argument("--tool", help=f"Name of the tool which calls {PROGRAM_NAME}. This name will be used to invoke appropriate API call, "
                                       f"which will then do additional work in tool-dependent way before running standard {PROGRAM_NAME}.")
    parser.add_argument("-V", help="Print version", action='version',
                        version=f"{PROGRAM_NAME} {VERSION}")

    namespace = parser.parse_args()
    namespace.specification_path = Path(namespace.specification_path)
    namespace.result_path = Path(namespace.result_path)
    cwd_dir = os.getcwd()

    hello_world(namespace.specification_path, cwd_dir, namespace.result_path)

    run_immuneML(namespace)
   
    # bye_bye_world()  # To activate once the paper is accepted.

if __name__ == "__main__":
    main()
