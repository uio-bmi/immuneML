import logging

from scripts.DocumentatonFormat import DocumentationFormat
from scripts.specification_util import write_class_docs
from source.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.logging.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.HPOptimizationInstruction import HPOptimizationInstruction
from source.workflows.instructions.SimulationInstruction import SimulationInstruction
from source.workflows.instructions.dataset_generation.DatasetGenerationInstruction import DatasetGenerationInstruction
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from source.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction
from source.workflows.instructions.ml_model_training.MLModelTrainingInstruction import MLModelTrainingInstruction


class InstructionParser:

    keyword = "instructions"

    @staticmethod
    def parse(definition_output: DefinitionParserOutput, path):

        specification = definition_output.specification
        symbol_table = definition_output.symbol_table

        if InstructionParser.keyword in specification:

            if len(specification[InstructionParser.keyword].keys()) > 1:
                logging.warning(f"InstructionParser: multiple instructions were listed in the specification (under keys "
                                f"{str(list(specification[InstructionParser.keyword].keys()))[1:-1]}). "
                                "These instructions are independent and results from one instruction are not available to others. "
                                "If this is the intended behavior, please ignore this warning. If the output of one instruction is needed for the "
                                "other, please use separate YAML specifications and separate runs to perform the analysis.")

            for key in specification[InstructionParser.keyword]:
                specification[InstructionParser.keyword][key], symbol_table = \
                    InstructionParser.parse_instruction(key, specification[InstructionParser.keyword][key], symbol_table, path)
        else:
            specification[InstructionParser.keyword] = {}

        return symbol_table, specification[InstructionParser.keyword]

    @staticmethod
    @log
    def parse_instruction(key: str, instruction: dict, symbol_table: SymbolTable, path) -> tuple:
        # TODO: add check to see if there's type
        valid_instructions = [cls[:-6] for cls in ReflectionHandler.discover_classes_by_partial_name("Parser", "dsl/instruction_parsers/")]
        ParameterValidator.assert_in_valid_list(instruction["type"], valid_instructions, "InstructionParser", "type")

        parser = ReflectionHandler.get_class_by_name("{}Parser".format(instruction["type"]), "instruction_parsers/")()
        instruction_object = parser.parse(key, instruction, symbol_table, path)

        symbol_table.add(key, SymbolType.INSTRUCTION, instruction_object)
        return instruction, symbol_table

    @staticmethod
    def generate_docs(path):
        inst_path = PathBuilder.build(f"{path}instructions/")
        InstructionParser.make_dataset_generation_docs(inst_path)
        InstructionParser.make_expl_analysis_docs(inst_path)
        InstructionParser.make_hp_docs(inst_path)
        InstructionParser.make_simulation_docs(inst_path)
        InstructionParser.make_ml_model_training_docs(inst_path)
        InstructionParser.make_ml_application_docs(inst_path)

    @staticmethod
    def make_ml_application_docs(path):
        with open(f"{path}ml_application.rst", "w") as file:
            write_class_docs(DocumentationFormat(MLApplicationInstruction, "", DocumentationFormat.LEVELS[1]), file)

    @staticmethod
    def make_ml_model_training_docs(path):
        with open(f"{path}ml_model_training.rst", "w") as file:
            write_class_docs(DocumentationFormat(MLModelTrainingInstruction, "", DocumentationFormat.LEVELS[1]), file)

    @staticmethod
    def make_dataset_generation_docs(path):
        with open(f"{path}dataset_generation.rst", "w") as file:
            write_class_docs(DocumentationFormat(DatasetGenerationInstruction, "", DocumentationFormat.LEVELS[1]), file)

    @staticmethod
    def make_expl_analysis_docs(path):
        with open(f"{path}exploratory_analysis.rst", "w") as file:
            write_class_docs(DocumentationFormat(ExploratoryAnalysisInstruction, "", DocumentationFormat.LEVELS[1]), file)

    @staticmethod
    def make_simulation_docs(path):
        with open(f"{path}simulation.rst", "w") as file:
            write_class_docs(DocumentationFormat(SimulationInstruction, "", DocumentationFormat.LEVELS[1]), file)

    @staticmethod
    def make_hp_docs(path):
        with open(f"{path}hp.rst", "w") as file:
            write_class_docs(DocumentationFormat(HPOptimizationInstruction, "", DocumentationFormat.LEVELS[1]), file)
            write_class_docs(DocumentationFormat(SplitConfig, "SplitConfig", DocumentationFormat.LEVELS[1]), file)
            write_class_docs(DocumentationFormat(ReportConfig, "ReportConfig", DocumentationFormat.LEVELS[1]), file)
