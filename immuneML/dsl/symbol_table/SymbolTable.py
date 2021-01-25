import warnings
from typing import List

from immuneML.dsl.symbol_table.SymbolTableEntry import SymbolTableEntry
from immuneML.dsl.symbol_table.SymbolType import SymbolType


class SymbolTable:

    """
    Symbol table contains all objects parsed from the specification in the following format:

    --------------------------------------------------------------------------------------------------
    symbol | symbol_type   | item                                                      | config      |
    --------------------------------------------------------------------------------------------------
    e1     | encoding      | EncodingObject                                            | {...}       | -> SymbolTableEntry object
    seq1   | preprocessing | [ClonesPerRepertoireFilter(), MetadataRepertoireFilter()] | {...}       | -> SymbolTableEntry object

    """

    def __init__(self):
        self._items = {}

    def add(self, symbol: str, symbol_type: SymbolType, item, config: dict = None):
        if symbol in self._items.keys() and self._items[symbol] is not None:
            warnings.warn("An item with the key {} was already set in the SymbolTable during parsing. If overwriting "
                          "it was the intended behavior, please ignore this warning.".format(symbol), Warning)

        self._items[symbol] = SymbolTableEntry(symbol=symbol, symbol_type=symbol_type, item=item, config=config)

    def get(self, symbol):
        if symbol is not None:
            if self.contains(symbol):
                return self._items[symbol].item
            else:
                raise KeyError(f"SymbolTable: item with key {symbol} was not defined previously so it could not be retrieved during "
                               f"parsing. Please check if an item with key {symbol} was defined in the specification.")
        else:
            return None

    def get_config(self, symbol):
        return self._items[symbol].config

    def contains(self, symbol):
        return symbol in self._items

    def get_by_type(self, symbol_type: SymbolType) -> List[SymbolTableEntry]:
        items = [self._items[key] for key in self._items.keys() if self._items[key].symbol_type == symbol_type]
        return items

    def get_keys_by_type(self, symbol_type: SymbolType) -> list:
        return [key for key in self._items.keys() if self._items[key].symbol_type == symbol_type]

    def __str__(self):
        return f"SymbolTable()"

    __repr__ = __str__
