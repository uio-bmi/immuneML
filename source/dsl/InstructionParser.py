from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


class InstructionParser:

    keyword = "instructions"

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        if InstructionParser.keyword in workflow_specification:
            for key in workflow_specification[InstructionParser.keyword]:
                workflow_specification[InstructionParser.keyword][key], symbol_table = \
                    InstructionParser.parse_instruction(workflow_specification[InstructionParser.keyword][key],
                                                        key,
                                                        symbol_table)
        else:
            workflow_specification[InstructionParser.keyword] = {}

        return symbol_table, workflow_specification[InstructionParser.keyword]

    @staticmethod
    def parse_instruction(instruction: dict, key: str, symbol_table: SymbolTable) -> tuple:
        parser = ReflectionHandler.get_class_by_name("{}Parser".format(key), "instruction_parsers/")()
        process = parser.parse(instruction, symbol_table)
        symbol_table.add(key, SymbolType.INSTRUCTION, process)
        return instruction, symbol_table
