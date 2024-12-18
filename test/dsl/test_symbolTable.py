from unittest import TestCase

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType


class TestSymbolTable(TestCase):
    def test_add(self):
        symbol_table = SymbolTable()
        symbol_table.add("svm1", SymbolType.ML_METHOD, {})
        with self.assertLogs():
            symbol_table.add("svm1", SymbolType.ML_METHOD, {})
