import logging

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType


class DefinitionParserOutput:

    def __init__(self, symbol_table: SymbolTable, specification: dict):
        if not any(len(symbol_table.get_by_type(symbol_type)) > 0 for symbol_type in [t for t in SymbolType]):
            logging.warning("DefinitionParserOutput: symbol table has not been populated by objects of any type."
                          "Please check if 'definitions' are correctly specified. ")

        self.symbol_table = symbol_table
        self.specification = specification
