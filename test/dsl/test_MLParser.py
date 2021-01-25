import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.Constants import Constants
from immuneML.ml_methods.LogisticRegression import LogisticRegression


class TestMLParser(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_parse_ml_methods(self):

        params = {
            "LR1": {
                "LogisticRegression": {
                    "max_iter": 1000,
                    "penalty": "l1",
                }
            },
            "LR2": "LogisticRegression",
            "SVM1": {
                "SVM": {
                    "max_iter": [1000, 2000],
                    "penalty": ["l1", "l2"]
                },
                "model_selection_cv": True,
                "model_selection_n_folds": 5
            },
            "SVM2": {
                "SVM": {},
                "model_selection_cv": False,
                "model_selection_n_folds": -1
            }
        }

        symbol_table = SymbolTable()
        symbol_table, desc = MLParser.parse(params, symbol_table)
        self.assertTrue(symbol_table.get("SVM1")._parameter_grid is not None and len(symbol_table.get("SVM1")._parameter_grid["max_iter"]) == 2)
        self.assertTrue(symbol_table.get("LR1")._parameters is not None and symbol_table.get("LR1")._parameters["penalty"] == "l1")
        self.assertTrue(isinstance(symbol_table.get("LR2"), LogisticRegression))

        self.assertTrue("SVM" in desc["SVM1"].keys())

    def test_check_keys(self):
        parsed_dict = {
            "a": {
                "b323_432": 3
            },
            "_a": 3,
            "sa": {
                "432!43": 2
            }
        }

        with self.assertRaises(AssertionError):
            ImmuneMLParser.check_keys(parsed_dict)

        parsed_dict2 = {
            "s": {
                "s": {
                    "fds": 3,
                    "324": {
                        "dsada": 2
                    }
                }
            }
        }

        ImmuneMLParser.check_keys(parsed_dict2)
