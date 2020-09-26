import logging
import os

from scripts.DocumentatonFormat import DocumentationFormat
from scripts.specification_util import write_class_docs
from source.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.util.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.Instruction import Instruction
from source.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


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
        instructions = sorted(ReflectionHandler.all_nonabstract_subclasses(Instruction, "Instruction", subdirectory='instructions/'), key=lambda x: x.__name__)

        inst_paths = {}

        for instruction in instructions:
            instruction_name = instruction.__name__[:-11]
            if hasattr(InstructionParser, f"make_{instruction_name}_docs"):
                fn = getattr(InstructionParser, f"make_{instruction_name.lower()}_docs")
                file_path = fn(inst_path)
            else:
                file_path = InstructionParser.make_docs(instruction, instruction_name, inst_path)

            inst_paths[instruction_name] = file_path

        print(inst_paths)

        with open(f'{inst_path}instructions.rst', 'w') as file:
            for key, item in inst_paths.items():
                lines = f"{key}\n---------------------------\n.. include:: {os.path.relpath(item, EnvironmentSettings.source_docs_path)}\n"
                print(lines)
                file.writelines(lines)

    @staticmethod
    def make_docs(instruction, name, path):
        with open(f"{path}{name}.rst", "w") as file:
            write_class_docs(DocumentationFormat(instruction, "", DocumentationFormat.LEVELS[1]), file)
        return f"{path}{name}.rst"

    @staticmethod
    def make_trainmlmodel_docs(path):
        with open(f"{path}hp.rst", "w") as file:
            write_class_docs(DocumentationFormat(TrainMLModelInstruction, "", DocumentationFormat.LEVELS[1]), file)
            write_class_docs(DocumentationFormat(SplitConfig, "SplitConfig", DocumentationFormat.LEVELS[1]), file)
            write_class_docs(DocumentationFormat(ReportConfig, "ReportConfig", DocumentationFormat.LEVELS[1]), file)
        return f"{path}hp.rst"
