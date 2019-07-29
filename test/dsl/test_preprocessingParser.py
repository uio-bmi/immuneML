from unittest import TestCase

from source.dsl.PreprocessingParser import PreprocessingParser
from source.dsl.SymbolTable import SymbolTable


class TestPreprocessingParser(TestCase):
    def test_parse(self):
        workflow_specs = {
            "preprocessing_sequences": {
                "seq1": [
                    {"filter_chain_B": {
                        "type": "DatasetChainFilter",
                        "params": {
                            "keep_chain": "A"
                        }
                    }}
                ],
                "seq2": [
                    {"filter_chain_A": {
                        "type": "DatasetChainFilter",
                        "params": {
                            "keep_chain": "B"
                        }
                    }}
                ]
            }
        }
        symbol_table = SymbolTable()
        table, specs = PreprocessingParser.parse(workflow_specs, symbol_table)

        self.assertTrue(table.contains("seq1"))
        self.assertTrue(table.contains("seq2"))
        self.assertTrue(isinstance(table.get("seq1"), list) and len(table.get("seq1")) == 1)
        self.assertEqual(list(workflow_specs["preprocessing_sequences"].keys()), list(specs.keys()))
