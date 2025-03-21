from inspect import signature
from pathlib import Path

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from immuneML.dsl.definition_parsers.EncodingParser import EncodingParser
from immuneML.dsl.definition_parsers.ExampleWeightingParser import ExampleWeightingParser
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.dsl.definition_parsers.MotifParser import MotifParser
from immuneML.dsl.definition_parsers.PreprocessingParser import PreprocessingParser
from immuneML.dsl.definition_parsers.ReportParser import ReportParser
from immuneML.dsl.definition_parsers.SignalParser import SignalParser
from immuneML.dsl.definition_parsers.SimulationParser import SimulationParser
from immuneML.dsl.import_parsers.ImportParser import ImportParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.reports.clustering_method_reports.ClusteringMethodReport import ClusteringMethodReport
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.reports.gen_model_reports.GenModelReport import GenModelReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.reports.multi_dataset_reports.MultiDatasetReport import MultiDatasetReport
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.implants.LigoPWM import LigoPWM
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.DocumentatonFormat import DocumentationFormat
from scripts.specification_util import write_class_docs, make_docs


class DefinitionParser:

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable, result_path: Path):

        specs = workflow_specification["definitions"]

        specs_defs = {}

        for parser in [MotifParser, SignalParser, SimulationParser, PreprocessingParser, EncodingParser, ExampleWeightingParser,
                       MLParser, ReportParser, ImportParser]:
            symbol_table, new_specs = DefinitionParser._call_if_exists(parser.keyword, parser.parse, specs,
                                                                       symbol_table, result_path)
            specs_defs[parser.keyword] = new_specs

        return DefinitionParserOutput(symbol_table=symbol_table, specification=workflow_specification), specs_defs

    @staticmethod
    def _call_if_exists(key: str, method, specs: dict, symbol_table: SymbolTable, path=None):
        if key in specs:
            if "path" in signature(method).parameters:
                return method(specs[key], symbol_table, path)
            else:
                return method(specs[key], symbol_table)
        else:
            return symbol_table, {}

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
        classes_to_document = [DocumentationFormat(Motif, "Motifs", DocumentationFormat.LEVELS[1]),
                               DocumentationFormat(SeedMotif, SeedMotif.__name__, DocumentationFormat.LEVELS[2]),
                               DocumentationFormat(LigoPWM, "PWM", DocumentationFormat.LEVELS[2]),
                               DocumentationFormat(Signal, "Signals", DocumentationFormat.LEVELS[1]),
                               DocumentationFormat(SimConfig, "Simulation config", DocumentationFormat.LEVELS[1]),
                               DocumentationFormat(SimConfigItem, "Simulation config item", DocumentationFormat.LEVELS[2])]

        file_path = path / "specs_simulation.rst"
        with file_path.open("w") as file:
            for doc_format in classes_to_document:
                write_class_docs(doc_format, file)

    @staticmethod
    def make_dataset_docs(path: Path):
        import_classes = ReflectionHandler.all_nonabstract_subclasses(DataImport, "Import", "dataset_import/")
        make_docs(path, import_classes, "specs_datasets.rst", "Import")

    @staticmethod
    def make_encodings_docs(path: Path):
        enc_classes = ReflectionHandler.all_direct_subclasses(DatasetEncoder, "Encoder", "encodings/")
        enc_classes = [cl for cl in enc_classes if cl.__name__ != "ProteinEmbeddingEncoder"]
        enc_classes += ReflectionHandler.all_direct_subclasses(ProteinEmbeddingEncoder, "Encoder", "encodings/")

        make_docs(path, enc_classes, "specs_encodings.rst", "Encoder")

    @staticmethod
    def make_reports_docs(path: Path):
        filename = "specs_reports.rst"
        file_path = path / filename
        mode = "w"

        report_types = [DataReport, EncodingReport, MLReport, GenModelReport, TrainMLModelReport, ClusteringReport,
                        MultiDatasetReport]

        for report_type_class in report_types:
            with file_path.open(mode) as file:
                doc_format = DocumentationFormat(cls=report_type_class,
                                                 cls_name=f"**{report_type_class.DOCS_TITLE}**",
                                                 level_heading=DocumentationFormat.LEVELS[1])
                write_class_docs(doc_format, file)
                mode = "a"

            subdir = DefaultParamsLoader.convert_to_snake_case(report_type_class.__name__) + "s"

            classes = ReflectionHandler.all_nonabstract_subclasses(report_type_class, "", f"reports/{subdir}/")
            make_docs(path, classes, filename, "", "a", format_level=DocumentationFormat.LEVELS[2])

    @staticmethod
    def make_ml_methods_docs(path: Path):
        filename = 'specs_ml_methods.rst'
        file_path = path / filename
        mode = "w"

        method_mapping = [{'method_type': MLMethod, 'subdir': 'classifiers', 'title': MLMethod.DOCS_TITLE},
                          {'method_type': ClusteringMethod, 'subdir': 'clustering', 'title': ClusteringMethod.DOCS_TITLE},
                          {'method_type': GenerativeModel, 'subdir': 'generative_models', 'title': GenerativeModel.DOCS_TITLE},
                          {'method_type': DimRedMethod, 'subdir': 'dim_reduction', 'title': DimRedMethod.DOCS_TITLE}]

        for method in method_mapping:
            with file_path.open(mode) as file:
                doc_format = DocumentationFormat(cls=method['method_type'],
                                                 cls_name=f"**{method['title']}**",
                                                 level_heading=DocumentationFormat.LEVELS[1])
                write_class_docs(doc_format, file)
                mode = "a"

            classes = ReflectionHandler.all_nonabstract_subclasses(method['method_type'], "",
                                                                   f"ml_methods/{method['subdir']}/")
            make_docs(path, classes, filename, "", "a", format_level=DocumentationFormat.LEVELS[2])

    @staticmethod
    def make_preprocessing_docs(path: Path):
        classes = ReflectionHandler.all_nonabstract_subclasses(Preprocessor, "", "preprocessing/")
        make_docs(path, classes, "specs_preprocessings.rst", "")
