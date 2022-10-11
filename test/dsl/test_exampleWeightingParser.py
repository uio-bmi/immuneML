import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.dsl.definition_parsers.ExampleWeightingParser import ExampleWeightingParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.Constants import Constants
from immuneML.example_weighting.importance_weighting.ImportanceWeighting import ImportanceWeighting
from immuneML.example_weighting.predefined_weighting.PredefinedWeighting import PredefinedWeighting


class TestExampleWeightingParser(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_parse_ml_methods(self):

        params = {
            "w1": {
                "ImportanceWeighting": {
                    "baseline_dist": "olga",
                    "dataset_dist": "mutagenesis",
                }
            },
            "w2": "ImportanceWeighting",
            "w3": {
                "PredefinedWeighting": {
                    "file_path": "example/path.csv"
                }
            }
        }

        symbol_table = SymbolTable()
        symbol_table, desc = ExampleWeightingParser.parse(params, symbol_table)

        self.assertEqual(ImportanceWeighting, symbol_table.get("w1"))
        self.assertEqual(ImportanceWeighting, symbol_table.get("w2"))
        self.assertEqual(PredefinedWeighting, symbol_table.get("w3"))

        self.assertEqual(symbol_table.get("w1"), ImportanceWeighting)
        self.assertEqual(symbol_table.get("w2"), ImportanceWeighting)
        self.assertEqual(symbol_table.get("w3"), PredefinedWeighting)
        self.assertEqual(symbol_table.get_config("w1"), {'example_weighting_params': {'baseline_dist': 'olga',
                                                                                      'dataset_dist': 'mutagenesis',
                                                                                      'name': 'w1'}})
        self.assertEqual(symbol_table.get_config("w2"), {'example_weighting_params': {'baseline_dist': 'uniform',
                                                                                      'dataset_dist': 'mutagenesis',
                                                                                      'name': 'w2'}})
        self.assertEqual(symbol_table.get_config("w3"), {'example_weighting_params': {'file_path': 'example/path.csv',
                                                                                      'name': 'w3'}})


