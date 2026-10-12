# quality: peripheral
import datetime
import re
from pathlib import Path

import yaml
from yaml import MarkedYAMLError

from immuneML.dsl.InstructionParser import InstructionParser
from immuneML.dsl.OutputParser import OutputParser
from immuneML.dsl.definition_parsers.DefinitionParser import DefinitionParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class ImmuneMLParser:
    """
    Simple DSL parser from python dictionary or equivalent YAML for configuring repertoire / receptor_sequence
    classification in the (simulated) settings
    """

    @staticmethod
    def parse_yaml_file(file_path: Path, result_path: Path = None, parse_func=None):
        try:
            with file_path.open("r") as file:
                workflow_specification = yaml.safe_load(file)
                ImmuneMLParser.check_keys(workflow_specification)
        except yaml.YAMLError as exc:
            problem_description = "\n--------------------------------------------------------------------------------\n" \
                                  "There was a YAML formatting error in the supplied specification file. Please validate specification " \
                                  "(you can use https://jsonformatter.org/yaml-validator) and try again."
            raise MarkedYAMLError(context=str(exc), problem=problem_description, problem_mark=f"The error was {exc.problem_mark}.")

        try:
            if parse_func is None:
                symbol_table, path = ImmuneMLParser.parse(workflow_specification, file_path, result_path)
            else:
                symbol_table, path = parse_func(workflow_specification, file_path, result_path)
        except KeyError as key_error:
            raise Exception(f"ImmuneMLParser: an error occurred during parsing the YAML specification. Missing key was '{key_error.args[0]}'. "
                            f"For more details, refer to the log above and check the documentation.") from key_error
        return symbol_table, path

    @staticmethod
    def check_keys(specs: dict):
        for key in specs.keys():
            key_to_check = str(key)
            assert re.match(r'^[A-Za-z0-9_]+$', key_to_check), \
                f"ImmuneMLParser: the keys in the specification can contain only letters, numbers and underscore. Error with key: {key}"
            if isinstance(specs[key], dict) and key not in ["column_mapping", "metadata_column_mapping"]:
                ImmuneMLParser.check_keys(specs[key])

    @staticmethod
    def parse(workflow_specification: dict, file_path, result_path):

        symbol_table = SymbolTable()

        def_parser_output, specs_defs = DefinitionParser.parse(workflow_specification, symbol_table, result_path)
        symbol_table, specs_instructions = InstructionParser.parse(def_parser_output, result_path)
        app_output = OutputParser.parse(workflow_specification, symbol_table)

        path = ImmuneMLParser._output_specs(file_path=file_path, result_path=result_path, definitions=specs_defs,
                                            instructions=specs_instructions, output=app_output,
                                            random_seed=workflow_specification.get("random_seed"))

        return symbol_table, path

    @staticmethod
    def _get_full_specs_filepath(file_path, result_path) -> Path:
        file_name = f"full_{file_path.stem}.yaml"
        if result_path is None:
            folder = file_path.absolute().parent
            return folder / file_name
        else:
            return result_path / file_name

    @staticmethod
    def _output_specs(file_path=None, result_path=None, definitions: dict = None, instructions: dict = None, output: dict = None,
                      random_seed: int = None) -> Path:
        filepath = ImmuneMLParser._get_full_specs_filepath(file_path, result_path)

        result = {"definitions": definitions, "instructions": instructions, "output": output}
        if random_seed is not None:
            result["random_seed"] = random_seed
        result = ImmuneMLParser._paths_to_strings_recursive(result)

        PathBuilder.build(filepath.parent)
        with filepath.open("w") as file:
            yaml.dump(result, file)

        print_log(f"Full specification is available at {filepath}.\n", include_datetime=True)
        return filepath

    @staticmethod
    def _paths_to_strings_recursive(specs):
        if isinstance(specs, Path):
            return specs.as_posix()
        elif isinstance(specs, dict):
            return {key: ImmuneMLParser._paths_to_strings_recursive(value) for key, value in specs.items()}
        elif isinstance(specs, list):
            return [ImmuneMLParser._paths_to_strings_recursive(item) for item in specs]
        else:
            return specs

