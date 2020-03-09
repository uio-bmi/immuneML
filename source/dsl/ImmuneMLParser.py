# quality: peripheral
import os
import re

import yaml

from source.dsl.InstructionParser import InstructionParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.definition_parsers.DefinitionParser import DefinitionParser


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
                    metadata: "./metadata.csv"
                    format: MiXCR
                    params:
                        result_path: "./loaded_dataset/"
                        sequence_type: CDR3
            encodings:
                e1:
                    KmerFrequency
                        k: 3
                e2:
                    Word2Vec
                        vector_size: 16
                        context: sequence
            ml_methods:
                log_reg1:
                    LogisticRegression
                        C: 0.001
            reports:
                r1:
                    SequenceLengthDistribution
            preprocessing_sequences:
                seq1:
                    - filter_chain_B:
                        DatasetChainFilter
                            keep_chain: A
                    - filter_clonotype:
                        ClonotypeCountFilter
                            lower_limit: 1000
                seq2:
                    - filter_clonotype:
                        ClonotypeCountFilter
                            lower_limit: 500
                    - filter_chain_A:
                        DatasetChainFilter
                            keep_chain: B
        instructions:
            inst1:
                type: HPOptimization
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
                        data_splits: []
                        performance: []
                        optimal_models: []
                selection:
                    split_strategy: k-fold
                    split_count: 5
                    reports:
                        data_splits: [r1]
                        models: []
                        data: []
                labels:
                    - CD
                dataset: d1
                strategy: GridSearch
                metrics: [accuracy, f1_micro]
                reports: None

    """

    @staticmethod
    def parse_yaml_file(file_path, result_path=None):
        try:
            with open(file_path, "r") as file:
                workflow_specification = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"YAML formatting error in the specification file. Validate the specification and try again.\n"
                  f"Error info: {exc}")
            raise exc

        try:
            symbol_table, path = ImmuneMLParser.parse(workflow_specification, file_path, result_path)
        except KeyError as key_error:
            raise Exception(f"ImmuneMLParser: an error occurred during parsing. Missing key was {key_error.args[0]}. "
                            f"For more details, refer to the log above and check the documentation.") from key_error
        return symbol_table, path

    @staticmethod
    def check_keys(specs: dict):
        for key in specs.keys():
            assert re.match(r'^[A-Za-z0-9_]+$', key), \
                f"ImmuneMLParser: the keys in the specification can contain only letters, numbers and underscore. Error with key: {key}"
            if isinstance(specs[key], dict):
                ImmuneMLParser.check_keys(specs[key])

    @staticmethod
    def parse(workflow_specification: dict, file_path, result_path):

        symbol_table = SymbolTable()

        def_parser_output, specs_defs = DefinitionParser.parse(workflow_specification, symbol_table)
        symbol_table, specs_instructions = InstructionParser.parse(def_parser_output)

        path = ImmuneMLParser._output_specs(file_path=file_path, result_path=result_path, definitions=specs_defs,
                                            instructions=specs_instructions)

        return symbol_table, path

    @staticmethod
    def _get_full_specs_filepath(file_path, result_path):
        file_name = os.path.basename(file_path).split(".")[0]
        file_name = "full_{}.yaml".format(file_name)
        if result_path is None:
            folder = os.path.dirname(os.path.abspath(file_path))
            return folder + "/" + file_name
        else:
            return result_path + file_name

    @staticmethod
    def _output_specs(file_path=None, result_path=None, definitions: dict = None, instructions: dict = None):
        filepath = ImmuneMLParser._get_full_specs_filepath(file_path, result_path)

        result = {"definitions": definitions, "instructions": instructions}
        with open(filepath, "w") as file:
            yaml.dump(result, file)

        print("Full specification is available at {}.".format(filepath))
        return filepath
