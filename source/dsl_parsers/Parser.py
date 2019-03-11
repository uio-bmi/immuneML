# quality: peripheral
import yaml

from source.dsl_parsers.EncodingParser import EncodingParser
from source.dsl_parsers.MLParser import MLParser
from source.dsl_parsers.ReportParser import ReportParser
from source.dsl_parsers.SimulationParser import SimulationParser


class Parser:
    """
    Simple DSL parser from python dictionary or equivalent YAML for configuring repertoire / receptor_sequence
    classification in the (simulated) settings
    """

    @staticmethod
    def parse_yaml_file(file_path) -> dict:
        with open(file_path, "r") as file:
            workflow_specification = yaml.load(file)

        return Parser.parse(workflow_specification)

    @staticmethod
    def parse(workflow_specification: dict) -> dict:
        result = {}
        if "simulation" in workflow_specification:
            result["simulation"], result["signals"] = SimulationParser.parse_simulation(workflow_specification["simulation"])
        if "ml_methods" in workflow_specification:
            result["ml_methods"] = MLParser.parse_ml_methods(workflow_specification["ml_methods"])
        if "encoder" in workflow_specification and "encoder_params" in workflow_specification:
            result["encoder"], result["encoder_params"] = EncodingParser.parse_encoder(workflow_specification)
        if "reports" in workflow_specification:
            result["reports"] = ReportParser.parse_reports(workflow_specification["reports"])

        for key in workflow_specification.keys():
            if key not in result.keys():
                result[key] = workflow_specification[key]

        if "batch_size" not in workflow_specification.keys():
            result["batch_size"] = 1  # TODO: read this from configuration file / config object or sth similar

        return result
