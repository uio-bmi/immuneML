from scripts.DocumentatonFormat import DocumentationFormat
from scripts.specification_util import write_class_docs, make_docs
from source.IO.dataset_import.DataImport import DataImport
from source.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from source.dsl.definition_parsers.EncodingParser import EncodingParser
from source.dsl.definition_parsers.MLParser import MLParser
from source.dsl.definition_parsers.MotifParser import MotifParser
from source.dsl.definition_parsers.PreprocessingParser import PreprocessingParser
from source.dsl.definition_parsers.ReportParser import ReportParser
from source.dsl.definition_parsers.SignalParser import SignalParser
from source.dsl.definition_parsers.SimulationParser import SimulationParser
from source.dsl.import_parsers.ImportParser import ImportParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.encodings.DatasetEncoder import DatasetEncoder
from source.ml_methods.MLMethod import MLMethod
from source.preprocessing.Preprocessor import Preprocessor
from source.reports.Report import Report
from source.simulation.Implanting import Implanting
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.util.PathBuilder import PathBuilder
from source.util.ReflectionHandler import ReflectionHandler


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

    @staticmethod
    def generate_docs(path):
        def_path = PathBuilder.build(f"{path}definitions/")
        DefinitionParser.make_dataset_docs(def_path)
        DefinitionParser.make_simulation_docs(def_path)
        DefinitionParser.make_encodings_docs(def_path)
        DefinitionParser.make_reports_docs(def_path)
        DefinitionParser.make_ml_methods_docs(def_path)
        DefinitionParser.make_preprocessing_docs(def_path)

    @staticmethod
    def make_simulation_docs(path):
        instantiations = ReflectionHandler.all_nonabstract_subclasses(MotifInstantiationStrategy, "Instantiation",
                                                                      "motif_instantiation_strategy/")

        instantiations = [DocumentationFormat(inst, inst.__name__.replace('Instantiation', ""), DocumentationFormat.LEVELS[2])
                          for inst in instantiations]

        classes_to_document = [DocumentationFormat(Motif, Motif.__name__, DocumentationFormat.LEVELS[1])] + instantiations + \
                              [DocumentationFormat(Signal, Signal.__name__, DocumentationFormat.LEVELS[1]),
                               DocumentationFormat(Implanting, Implanting.__name__, DocumentationFormat.LEVELS[1])]

        with open(path + "simulation.rst", "w") as file:
            for doc_format in classes_to_document:
                write_class_docs(doc_format, file)

    @staticmethod
    def make_dataset_docs(path):
        import_classes = ReflectionHandler.all_nonabstract_subclasses(DataImport, "Import", "dataset_import/")
        make_docs(path, import_classes, "datasets.rst", "Import")

    @staticmethod
    def make_encodings_docs(path):
        enc_classes = ReflectionHandler.all_direct_subclasses(DatasetEncoder, "Encoder", "encodings/")
        make_docs(path, enc_classes, "encodings.rst", "Encoder")

    @staticmethod
    def make_reports_docs(path):
        classes = ReflectionHandler.all_nonabstract_subclasses(Report, "", "reports/")
        make_docs(path, classes, "reports.rst", "")

    @staticmethod
    def make_ml_methods_docs(path):
        classes = ReflectionHandler.all_nonabstract_subclasses(MLMethod, "", "ml_methods/")
        make_docs(path, classes, "ml_methods.rst", "")

    @staticmethod
    def make_preprocessing_docs(path):
        classes = ReflectionHandler.all_nonabstract_subclasses(Preprocessor, "", "preprocessing/")
        make_docs(path, classes, "preprocessings.rst", "")
