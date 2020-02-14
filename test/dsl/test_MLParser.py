from unittest import TestCase

from source.dsl.MLParser import MLParser
from source.dsl.SymbolTable import SymbolTable
from source.ml_methods.LogisticRegression import LogisticRegression


class TestMLParser(TestCase):
    def test_parse_ml_methods(self):

        params = {
            "ml_methods": {
                "LR1": {
                    "type": "LogisticRegression",
                    "params": {
                        "max_iter": 1000,
                        "penalty": "l1",
                    },
                    "encoding": "e1",
                    "labels": ["CD"],
                    "metrics": ["accuracy", "balanced_accuracy"],
                    "min_example_count": 1
                },
                "LR2": {
                    "type": "LogisticRegression",
                    "encoding": "e1",
                    "labels": ["CD"],
                    "metrics": ["accuracy", "balanced_accuracy"],
                    "min_example_count": 1
                },
                "SVM1": {
                    "type": "SVM",
                    "params": {
                        "max_iter": [1000, 2000],
                        "penalty": ["l1", "l2"]
                    },
                    "encoding": "e1",
                    "labels": ["CD"],
                    "metrics": ["accuracy", "balanced_accuracy"],
                    "split_count": 1,
                    "model_selection_cv": False,
                    "model_selection_n_folds": -1,
                    "assessment_type": "LOOCV",
                }
            }
        }

        symbol_table = SymbolTable()
        symbol_table, desc = MLParser.parse(params, symbol_table)
        self.assertTrue(symbol_table.get("SVM1")._parameter_grid is not None and len(symbol_table.get("SVM1")._parameter_grid["max_iter"]) == 2)
        self.assertTrue(symbol_table.get("LR1")._parameters is not None and symbol_table.get("LR1")._parameters["penalty"] == "l1")
        self.assertTrue(isinstance(symbol_table.get("LR2"), LogisticRegression))

        self.assertEqual("SVM", desc["SVM1"]["type"])
