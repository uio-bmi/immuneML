from source.dsl.definition_parsers.DefinitionParserOutput import DefinitionParserOutput
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.logging.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class InstructionParser:

    keyword = "instructions"

    @staticmethod
    def parse(definition_output: DefinitionParserOutput):

        specification = definition_output.specification
        symbol_table = definition_output.symbol_table

        if InstructionParser.keyword in specification:
            for key in specification[InstructionParser.keyword]:
                specification[InstructionParser.keyword][key], symbol_table = \
                    InstructionParser.parse_instruction(key, specification[InstructionParser.keyword][key], symbol_table)
        else:
            specification[InstructionParser.keyword] = {}

        return symbol_table, specification[InstructionParser.keyword]

    @staticmethod
    @log
    def parse_instruction(key: str, instruction: dict, symbol_table: SymbolTable) -> tuple:
        # TODO: add check to see if there's type
        valid_instructions = [cls[:-6] for cls in ReflectionHandler.discover_classes_by_partial_name("Parser", "dsl/instruction_parsers/")]
        ParameterValidator.assert_in_valid_list(instruction["type"], valid_instructions, "InstructionParser", "type")

        parser = ReflectionHandler.get_class_by_name("{}Parser".format(instruction["type"]), "instruction_parsers/")()
        instruction_object = parser.parse(key, instruction, symbol_table)

        symbol_table.add(key, SymbolType.INSTRUCTION, instruction_object)
        return instruction, symbol_table
