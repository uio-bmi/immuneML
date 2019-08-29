from source.dsl.DefinitionParserOutput import DefinitionParserOutput
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
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
                    InstructionParser.parse_instruction(specification[InstructionParser.keyword][key],
                                                        key,
                                                        symbol_table)
        else:
            specification[InstructionParser.keyword] = {}

        return symbol_table, specification[InstructionParser.keyword]

    @staticmethod
    def parse_instruction(instruction: dict, key: str, symbol_table: SymbolTable) -> tuple:
        parser = ReflectionHandler.get_class_by_name("{}Parser".format(instruction["type"]), "instruction_parsers/")()
        process = parser.parse(instruction, symbol_table)
        symbol_table.add(key, SymbolType.INSTRUCTION, process)
        return instruction, symbol_table
