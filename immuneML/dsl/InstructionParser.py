import logging
import os
from pathlib import Path

from immuneML.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction
from scripts.DocumentatonFormat import DocumentationFormat
from scripts.specification_util import write_class_docs


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
        ParameterValidator.assert_keys_present(list(instruction.keys()), ["type"], InstructionParser.__name__, key)
        valid_instructions = [cls[:-6] for cls in ReflectionHandler.discover_classes_by_partial_name("Parser", "dsl/instruction_parsers/")]
        ParameterValidator.assert_in_valid_list(instruction["type"], valid_instructions, "InstructionParser", "type")

        parser = ReflectionHandler.get_class_by_name("{}Parser".format(instruction["type"]), "instruction_parsers/")()
        instruction_object = parser.parse(key, instruction, symbol_table, path)

        symbol_table.add(key, SymbolType.INSTRUCTION, instruction_object)
        return instruction, symbol_table

    @staticmethod
    def generate_docs(path: Path):
        inst_path = PathBuilder.build(path / "instructions")
        instructions = sorted(ReflectionHandler.all_nonabstract_subclasses(Instruction, "Instruction", subdirectory='instructions/'), key=lambda x: x.__name__)

        inst_paths = {}

        for instruction in instructions:
            instruction_name = instruction.__name__[:-11]
            if hasattr(InstructionParser, f"make_{instruction_name.lower()}_docs"):
                fn = getattr(InstructionParser, f"make_{instruction_name.lower()}_docs")
                file_path = fn(inst_path)
            else:
                file_path = InstructionParser.make_docs(instruction, instruction_name, inst_path)

            inst_paths[instruction_name] = file_path

        inst_file_path = inst_path / "instructions.rst"
        with inst_file_path.open('w') as file:
            for key, item in inst_paths.items():
                lines = f"{key}\n---------------------------\n.. include:: {os.path.relpath(item, EnvironmentSettings.source_docs_path)}\n"
                file.writelines(lines)

    @staticmethod
    def make_docs(instruction, name, path: Path):
        file_path = path / f"{name}.rst"

        with file_path.open("w") as file:
            write_class_docs(DocumentationFormat(instruction, "", DocumentationFormat.LEVELS[1]), file)
        return file_path

    @staticmethod
    def make_trainmlmodel_docs(path):
        file_path = path / "hp.rst"
        with file_path.open("w") as file:
            write_class_docs(DocumentationFormat(TrainMLModelInstruction, "", DocumentationFormat.LEVELS[1]), file)
            write_class_docs(DocumentationFormat(SplitConfig, "SplitConfig", DocumentationFormat.LEVELS[1]), file)
            write_class_docs(DocumentationFormat(ReportConfig, "ReportConfig", DocumentationFormat.LEVELS[1]), file)
        return file_path
