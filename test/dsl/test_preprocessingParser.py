from unittest import TestCase

from immuneML.dsl.definition_parsers.PreprocessingParser import PreprocessingParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable


class TestPreprocessingParser(TestCase):
    def test_parse(self):
        workflow_specs = {
            "seq1": [
                {"filter_chain_B": {
                    "ChainRepertoireFilter": {
                        "keep_chains": ["A"]
                    }
                }}
            ],
            "seq2": [
                {"filter_chain_A": {
                    "ChainRepertoireFilter": {
                        "keep_chains": ["B"]
                    }
                }}
            ]
        }
        symbol_table = SymbolTable()
        table, specs = PreprocessingParser.parse(workflow_specs, symbol_table)

        self.assertTrue(table.contains("seq1"))
        self.assertTrue(table.contains("seq2"))
        self.assertTrue(isinstance(table.get("seq1"), list) and len(table.get("seq1")) == 1)
        self.assertEqual(list(workflow_specs.keys()), list(specs.keys()))
