from source.dsl.SymbolTable import SymbolTable
from source.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from source.dsl.definition_parsers.EncodingParser import EncodingParser
from source.dsl.definition_parsers.MLParser import MLParser
from source.dsl.definition_parsers.MotifParser import MotifParser
from source.dsl.definition_parsers.PreprocessingParser import PreprocessingParser
from source.dsl.definition_parsers.ReportParser import ReportParser
from source.dsl.definition_parsers.SignalParser import SignalParser
from source.dsl.definition_parsers.SimulationParser import SimulationParser
from source.dsl.import_parsers.ImportParser import ImportParser


class DefinitionParser:
    # TODO: remove redundancy from there, make lists and call those automatically instead

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):

        specs = workflow_specification["definitions"]

        symbol_table, specs_import = ImportParser.parse(specs, symbol_table)
        symbol_table, specs_motifs = DefinitionParser._call_if_exists("motifs", MotifParser.parse_motifs, specs, symbol_table)
        symbol_table, specs_signals = DefinitionParser._call_if_exists("signals", SignalParser.parse_signals, specs, symbol_table)
        symbol_table, specs_simulation = DefinitionParser._call_if_exists("simulations", SimulationParser.parse_simulations, specs, symbol_table)
        symbol_table, specs_preprocessing = DefinitionParser._call_if_exists(PreprocessingParser.keyword, PreprocessingParser.parse, specs, symbol_table)
        symbol_table, specs_encoding = DefinitionParser._call_if_exists("encodings", EncodingParser.parse, specs, symbol_table)
        symbol_table, specs_ml = DefinitionParser._call_if_exists("ml_methods", MLParser.parse, specs, symbol_table)
        symbol_table, specs_report = DefinitionParser._call_if_exists("reports", ReportParser.parse_reports, specs, symbol_table)

        specs_defs = DefinitionParser.create_specs_defs(specs_import, specs_simulation, specs_preprocessing,
                                                        specs_encoding, specs_ml, specs_report)

        return DefinitionParserOutput(symbol_table=symbol_table, specification=workflow_specification), specs_defs

    @staticmethod
    def _call_if_exists(key: str, method, specs: dict, symbol_table: SymbolTable):
        if key in specs:
            return method(specs[key], symbol_table)
        else:
            return symbol_table, {}

    @staticmethod
    def create_specs_defs(specs_datasets: dict, simulation: dict, preprocessings: dict,
                          encodings: dict, ml_methods: dict, reports: dict):

        return {
            "datasets": specs_datasets, "simulation": simulation, "preprocessings": preprocessings,
            "encodings": encodings, "ml_methods": ml_methods, "reports": reports
        }
