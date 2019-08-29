from source.dsl.MLParser import MLParser
from source.dsl.PreprocessingParser import PreprocessingParser
from source.dsl.ReportParser import ReportParser
from source.dsl.SimulationParser import SimulationParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.encoding_parsers.EncodingParser import EncodingParser
from source.dsl.import_parsers.ImportParser import ImportParser


class DefinitionParser:

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):

        specs = workflow_specification["definitions"]

        symbol_table, specs_import = ImportParser.parse(specs, symbol_table)
        symbol_table, specs_simulation = SimulationParser.parse_simulation(specs, symbol_table)
        symbol_table, specs_preprocessing = PreprocessingParser.parse(specs, symbol_table)
        symbol_table, specs_encoding = EncodingParser.parse(specs, symbol_table)
        symbol_table, specs_ml = MLParser.parse(specs, symbol_table)
        symbol_table, specs_report = ReportParser.parse_reports(specs, symbol_table)

        specs_defs = DefinitionParser.create_specs_defs(specs_import, specs_simulation, specs_preprocessing,
                                                        specs_encoding, specs_ml, specs_report)

        return symbol_table, specs_defs

    @staticmethod
    def create_specs_defs(specs_datasets: dict, simulation: dict, preprocessings: dict,
                          encodings: dict, ml_methods: dict, reports: dict):

        return {
            "datasets": specs_datasets, "simulation": simulation, "preprocessings": preprocessings,
            "encodings": encodings, "ml_methods": ml_methods, "reports": reports
        }
