from source.dsl.SymbolType import SymbolType


class SymbolTableEntry:

    def __init__(self, symbol: str, symbol_type: SymbolType, item, config: dict = None):
        self.symbol = symbol
        self.symbol_type = symbol_type
        self.item = item
        self.config = config
