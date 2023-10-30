import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.dsl.definition_parsers.ExampleWeightingParser import ExampleWeightingParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.Constants import Constants
from immuneML.example_weighting.predefined_weighting.PredefinedWeighting import PredefinedWeighting


class TestExampleWeightingParser(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_parse_example_weightings(self):

        params = {
            "w3": {
                "PredefinedWeighting": {
                    "file_path": "example/path.csv"
                }
            }
        }

        symbol_table = SymbolTable()
        symbol_table, desc = ExampleWeightingParser.parse(params, symbol_table)

        self.assertEqual(PredefinedWeighting, symbol_table.get("w3"))

        self.assertEqual(symbol_table.get("w3"), PredefinedWeighting)
        self.assertEqual(symbol_table.get_config("w3"), {'example_weighting_params': {'file_path': 'example/path.csv',
                                                                                      'separator': '\t',
                                                                                      'name': 'w3'}})


