import warnings

from source.dsl.SymbolTableEntry import SymbolTableEntry
from source.dsl.SymbolType import SymbolType


class SymbolTable:

    """
    Symbol table contains all objects parsed from the specification in the following format:

    ---------------------------------------------------------------------------------
    symbol | symbol_type   | item                                       | config    |
    ---------------------------------------------------------------------------------
    e1     | encoding      | EncodingObject                             | {...}     | -> SymbolTableEntry object
    seq1   | preprocessing | [ClonotypeCountFilter(), MetadataFilter()] | {...}     | -> SymbolTableEntry object

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
            return self._items[symbol].item
        else:
            return None

    def get_config(self, symbol):
        return self._items[symbol].config

    def contains(self, symbol):
        return symbol in self._items

    def get_by_type(self, symbol_type: SymbolType) -> list:
        items = [self._items[key] for key in self._items.keys() if self._items[key].symbol_type == symbol_type]
        return items
