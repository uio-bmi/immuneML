from unittest import TestCase

from source.dsl_parsers.MLParser import MLParser
from source.ml_methods.MLMethod import MLMethod


class TestMLParser(TestCase):
    def test_parse_ml_methods(self):

        params = {
            "ml_methods": {
                "LogisticRegression": {
                    "max_iter": 1000,
                    "penalty": ["l1", "l2"]
                },
                "SVM": {
                    "max_iter": [1000, 2000],
                    "penalty": ["l1", "l2"]
                }
            }
        }

        methods = MLParser.parse_ml_methods(params["ml_methods"])
        self.assertTrue(methods[0]._parameter_grid is not None and len(methods[0]._parameter_grid["max_iter"]) == 1)
        self.assertTrue(methods[1]._parameter_grid is not None and len(methods[1]._parameter_grid["max_iter"]) == 2)

        params = ["LogisticRegression", "RandomForestClassifier"]
        methods = MLParser.parse_ml_methods(params)
        self.assertTrue(all([isinstance(item, MLMethod) for item in methods]))
