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
from immuneML.util.PathBuilder import PathBuilder


class ImmuneMLParser:
    """
    Simple DSL parser from python dictionary or equivalent YAML for configuring repertoire / receptor_sequence
    classification in the (simulated) settings

    DSL example with hyper-parameter optimization:

    .. highlight:: yaml
    .. code-block:: yaml

        definitions:
            datasets:
                d1:
                    format: MiXCR
                    params:
                        result_path: loaded_dataset/
                        region_type: IMGT_CDR3
                        path: path_to_files/
                        metadata_file: metadata.csv
            encodings:
                e1:
                    KmerFrequency
                        k: 3
                e2:
                    Word2Vec:
                        vector_size: 16
                        context: sequence
            ml_methods:
                log_reg1:
                    LogisticRegression:
                        C: 0.001
            reports:
                r1:
                    SequenceLengthDistribution
            preprocessing_sequences:
                seq1:
                    - filter_chain_B:
                        ChainRepertoireFilter:
                            keep_chain: A
                    - filter_clonotype:
                        ClonesPerRepertoireFilter:
                            lower_limit: 1000
                seq2:
                    - filter_clonotype:
                        ClonesPerRepertoireFilter:
                            lower_limit: 500
                    - filter_chain_A:
                        ChainRepertoireFilter:
                            keep_chain: B
        instructions:
            inst1:
                type: TrainMLModel
                settings:
                    -   preprocessing: seq1
                        encoding: e1
                        ml_method: log_reg1
                    -   preprocessing: seq2
                        encoding: e2
                        ml_method: log_reg1
                assessment:
                    split_strategy: random
                    split_count: 1
                    training_percentage: 70
                    reports:
                        data: []
                        data_splits: []
                        encoding: []
                        models: []
                selection:
                    split_strategy: k-fold
                    split_count: 5
                    reports:
                        data: []
                        data_splits: [r1]
                        encoding: []
                        models: []
                labels:
                    - CD
                dataset: d1
                strategy: GridSearch
                metrics: [accuracy, f1_micro]
                optimization_metric: balanced_accuracy
                reports: []
        output: # this section can also be omitted, in that case output will be automatically HTML
            format: HTML # or None

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
            raise Exception(f"ImmuneMLParser: an error occurred during parsing. Missing key was {key_error.args[0]}. "
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
                                            instructions=specs_instructions, output=app_output)

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
    def _output_specs(file_path=None, result_path=None, definitions: dict = None, instructions: dict = None, output: dict = None) -> Path:
        filepath = ImmuneMLParser._get_full_specs_filepath(file_path, result_path)

        result = {"definitions": definitions, "instructions": instructions, "output": output}
        result = ImmuneMLParser._paths_to_strings_recursive(result)

        PathBuilder.build(filepath.parent)
        with filepath.open("w") as file:
            yaml.dump(result, file)

        print(f"{datetime.datetime.now()}: Full specification is available at {filepath}.\n", flush=True)
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

