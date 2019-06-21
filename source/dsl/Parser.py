# quality: peripheral
import os

import yaml

from source.dsl.import_parsers.ImportParser import ImportParser
from source.dsl.MLParser import MLParser
from source.dsl.ReportParser import ReportParser
from source.dsl.SimulationParser import SimulationParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.encoding_parsers.EncodingParser import EncodingParser


class Parser:
    """
    Simple DSL parser from python dictionary or equivalent YAML for configuring repertoire / receptor_sequence
    classification in the (simulated) settings
    """

    @staticmethod
    def parse_yaml_file(file_path, result_path=None):
        with open(file_path, "r") as file:
            workflow_specification = yaml.load(file)

        return Parser.parse(workflow_specification, file_path, result_path)

    @staticmethod
    def parse(workflow_specification: dict, file_path, result_path):

        symbol_table = SymbolTable()

        symbol_table, specs_import = ImportParser.parse(workflow_specification, symbol_table)
        symbol_table, specs_simulation = SimulationParser.parse_simulation(workflow_specification, symbol_table)
        symbol_table, specs_encoding = EncodingParser.parse(workflow_specification, symbol_table)
        symbol_table, specs_ml = MLParser.parse(workflow_specification, symbol_table)
        symbol_table, specs_report = ReportParser.parse_reports(workflow_specification, symbol_table)

        path = Parser._output_specs(file_path, result_path, specs_import, specs_simulation, specs_encoding, specs_ml, specs_report)

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
    def _output_specs(file_path=None, result_path=None, datasets: dict = None, simulation: dict = None,
                      encodings: dict = None, ml_methods: dict = None, reports: dict = None):
        filepath = Parser._get_full_specs_filepath(file_path, result_path)
        items = [datasets, simulation, encodings, ml_methods, reports]
        names = ["datasets", "simulation", "encodings", "ml_methods", "reports"]
        result = {names[index]: item for index, item in enumerate(items) if item is not None}
        with open(filepath, "w") as file:
            yaml.dump(result, file)

        print("Full specification is available at {}.".format(filepath))
        return filepath
