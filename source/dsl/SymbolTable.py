import warnings

from source.dsl.SymbolType import SymbolType


class SymbolTable:

    def __init__(self):
        self._items = {}

    def add(self, symbol: str, symbol_type: SymbolType, item: dict):
        if symbol in self._items.keys() and self._items[symbol] is not None:
            warnings.warn("An item with the key {} was already set in the SymbolTable during parsing. If overwriting "
                          "it was the intended behavior, please ignore this warning.".format(symbol), Warning)

        self._items[symbol] = item
        self._items[symbol]["type"] = symbol_type

    def get(self, symbol) -> dict:
        return self._items[symbol]

    def contains(self, symbol):
        return symbol in self._items.keys()

    def get_by_type(self, symbol_type: SymbolType) -> list:
        items = [(key, self._items[key]) for key in self._items.keys() if self._items[key]["type"] == symbol_type]
        return items
