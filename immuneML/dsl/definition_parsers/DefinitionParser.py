from pathlib import Path

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from immuneML.dsl.definition_parsers.EncodingParser import EncodingParser
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.dsl.definition_parsers.MotifParser import MotifParser
from immuneML.dsl.definition_parsers.PreprocessingParser import PreprocessingParser
from immuneML.dsl.definition_parsers.ReportParser import ReportParser
from immuneML.dsl.definition_parsers.SignalParser import SignalParser
from immuneML.dsl.definition_parsers.SimulationParser import SimulationParser
from immuneML.dsl.import_parsers.ImportParser import ImportParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.reports.multi_dataset_reports.MultiDatasetReport import MultiDatasetReport
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.simulation.Implanting import Implanting
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.DocumentatonFormat import DocumentationFormat
from scripts.specification_util import write_class_docs, make_docs


class DefinitionParser:

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable, result_path: Path):

        specs = workflow_specification["definitions"]

        symbol_table, specs_motifs = DefinitionParser._call_if_exists("motifs", MotifParser.parse_motifs, specs, symbol_table)
        symbol_table, specs_signals = DefinitionParser._call_if_exists("signals", SignalParser.parse_signals, specs, symbol_table)
        symbol_table, specs_simulation = DefinitionParser._call_if_exists("simulations", SimulationParser.parse_simulations, specs, symbol_table)
        symbol_table, specs_preprocessing = DefinitionParser._call_if_exists(PreprocessingParser.keyword, PreprocessingParser.parse, specs, symbol_table)
        symbol_table, specs_encoding = DefinitionParser._call_if_exists("encodings", EncodingParser.parse, specs, symbol_table)
        symbol_table, specs_ml = DefinitionParser._call_if_exists("ml_methods", MLParser.parse, specs, symbol_table)
        symbol_table, specs_report = DefinitionParser._call_if_exists("reports", ReportParser.parse_reports, specs, symbol_table)
        symbol_table, specs_import = ImportParser.parse(specs, symbol_table, result_path)

        specs_defs = DefinitionParser.create_specs_defs(specs_import, specs_simulation, specs_preprocessing, specs_motifs, specs_signals,
                                                        specs_encoding, specs_ml, specs_report)

        return DefinitionParserOutput(symbol_table=symbol_table, specification=workflow_specification), specs_defs

    @staticmethod
    def _call_if_exists(key: str, method, specs: dict, symbol_table: SymbolTable):
        if key in specs:
            return method(specs[key], symbol_table)
        else:
            return symbol_table, {}

    @staticmethod
    def create_specs_defs(specs_datasets: dict, simulation: dict, preprocessings: dict, motifs: dict, signals: dict,
                          encodings: dict, ml_methods: dict, reports: dict):

        return {
            "datasets": specs_datasets, "simulations": simulation, PreprocessingParser.keyword: preprocessings, "motifs": motifs, "signals": signals,
            "encodings": encodings, "ml_methods": ml_methods, "reports": reports
        }

    @staticmethod
    def generate_docs(path: Path):
        def_path = PathBuilder.build(path / "definitions")
        DefinitionParser.make_dataset_docs(def_path)
        DefinitionParser.make_simulation_docs(def_path)
        DefinitionParser.make_encodings_docs(def_path)
        DefinitionParser.make_reports_docs(def_path)
        DefinitionParser.make_ml_methods_docs(def_path)
        DefinitionParser.make_preprocessing_docs(def_path)

    @staticmethod
    def make_simulation_docs(path: Path):
        instantiations = ReflectionHandler.all_nonabstract_subclasses(MotifInstantiationStrategy, "Instantiation", "motif_instantiation_strategy/")
        instantiations = [DocumentationFormat(inst, inst.__name__.replace('Instantiation', ""), DocumentationFormat.LEVELS[2])
                          for inst in instantiations]

        implanting_strategies = ReflectionHandler.all_nonabstract_subclasses(SignalImplantingStrategy, 'Implanting', 'signal_implanting_strategy/')
        implanting_strategies = [DocumentationFormat(implanting, implanting.__name__.replace('Implanting', ""), DocumentationFormat.LEVELS[2])
                                 for implanting in implanting_strategies]

        classes_to_document = [DocumentationFormat(Motif, Motif.__name__, DocumentationFormat.LEVELS[1])] + instantiations + \
                              [DocumentationFormat(Signal, Signal.__name__, DocumentationFormat.LEVELS[1])] + implanting_strategies + \
                               [DocumentationFormat(Implanting, Implanting.__name__, DocumentationFormat.LEVELS[1])]

        file_path = path / "simulation.rst"
        with file_path.open("w") as file:
            for doc_format in classes_to_document:
                write_class_docs(doc_format, file)

    @staticmethod
    def make_dataset_docs(path: Path):
        import_classes = ReflectionHandler.all_nonabstract_subclasses(DataImport, "Import", "dataset_import/")
        make_docs(path, import_classes, "datasets.rst", "Import")

    @staticmethod
    def make_encodings_docs(path: Path):
        enc_classes = ReflectionHandler.all_direct_subclasses(DatasetEncoder, "Encoder", "encodings/")
        make_docs(path, enc_classes, "encodings.rst", "Encoder")

    @staticmethod
    def make_reports_docs(path: Path):
        filename = "reports.rst"
        file_path = path / filename

        with file_path.open("w") as file:
            pass

        for report_type_class in [DataReport, EncodingReport, MLReport, TrainMLModelReport, MultiDatasetReport]:
            with file_path.open("a") as file:
                doc_format = DocumentationFormat(cls=report_type_class,
                                                 cls_name=f"**{report_type_class.get_title()}**",
                                                 level_heading=DocumentationFormat.LEVELS[1])
                write_class_docs(doc_format, file)

            subdir = DefaultParamsLoader.convert_to_snake_case(report_type_class.__name__) + "s"

            classes = ReflectionHandler.all_nonabstract_subclasses(report_type_class, "", f"reports/{subdir}/")
            make_docs(path, classes, filename, "", "a")

    @staticmethod
    def make_ml_methods_docs(path: Path):
        classes = ReflectionHandler.all_nonabstract_subclasses(MLMethod, "", "ml_methods/")
        make_docs(path, classes, "ml_methods.rst", "")

    @staticmethod
    def make_preprocessing_docs(path: Path):
        classes = ReflectionHandler.all_nonabstract_subclasses(Preprocessor, "", "preprocessing/")
        make_docs(path, classes, "preprocessings.rst", "")
