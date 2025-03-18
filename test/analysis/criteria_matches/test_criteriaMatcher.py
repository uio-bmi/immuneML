from unittest import TestCase

import numpy as np
import pandas as pd

from immuneML.analysis.criteria_matches.BooleanType import BooleanType
from immuneML.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from immuneML.analysis.criteria_matches.OperationType import OperationType


class TestCriteriaMatcher(TestCase):

    def get_test_df(self):
        return pd.DataFrame({"matching_specificity": ["flu", "ebv", "GAD", "PPI", "GAD", "PPI", "ebv"],
                             "p_val": [0.01, 0.00001, 0.000001, 0.01, 0.01, 0.0000001, 0.1],
                             "odds_ratio": [0.51, 0.5, 0, 0, 0, 0, 0],
                             "a": ["yes", "no", "no", "no", "no", "no", "no"],
                             "b": ["no", "yes", "no", "no", "no", "no", "no"]})

    def test_match(self):
        df = self.get_test_df()

        filter_params = {
            "type": BooleanType.OR,
            "operands": [
                {
                    "type": BooleanType.AND,
                    "operands": [
                        {
                            "type": OperationType.IN,
                            "values": ["GAD", "PPI"],
                            "column": "matching_specificity"
                        },
                        {
                            "type": OperationType.LESS_THAN,
                            "threshold": 0.001,
                            "column": "p_val"
                        },
                    ]
                },
                {
                    "type": BooleanType.AND,
                    "operands": [
                        {
                            "type": OperationType.IN,
                            "values": ["yes"],
                            "column": "a"
                        },
                        {
                            "type": OperationType.GREATER_THAN,
                            "threshold": 0.5,
                            "column": "odds_ratio"
                        },
                    ]
                },
            ]
        }

        matcher = CriteriaMatcher()
        result = matcher.match(filter_params, df)
        self.assertTrue(np.array_equal(result, np.array([True, False, True, False, False, True, False])))

    def test_match_simple(self):
        df = self.get_test_df()

        filter_params = {
            "type": OperationType.TOP_N,
            "column": "odds_ratio",
            "number": 3
        }

        matcher = CriteriaMatcher()
        result = matcher.match(filter_params, df)
        self.assertTrue(np.array_equal(result, np.array([True, True, False, False, False, False, True])))

    def test_match_none(self):
        df = self.get_test_df()

        filter_params = {
            "type": OperationType.IN,
            "column": "matching_specificity",
            "values": ["GAD", "PPI"]
        }

        matcher = CriteriaMatcher()
        result = matcher.match(filter_params, df)
        self.assertTrue(np.array_equal(result, np.array([False, False, True, True, True, True, False])))
