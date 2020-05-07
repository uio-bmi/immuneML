import os
from unittest import TestCase

import numpy as np
import pandas as pd

from source.analysis.criteria_matches.BooleanType import BooleanType
from source.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants


class TestCriteriaMatcher(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_match(self):

        df = pd.DataFrame({"matching_specificity": ["flu", "ebv", "GAD", "PPI", "GAD", "PPI", "ebv"],
                           "p_val": [0.01, 0.00001, 0.000001, 0.01, 0.01, 0.0000001, 0.1],
                           "odds_ratio": [0.51, 0, 0, 0, 0, 0, 0],
                           "a": ["yes", "no", "no", "no", "no", "no", "no"],
                           "b": ["no", "yes", "no", "no", "no", "no", "no"]})

        filter_params = {
            "type": BooleanType.OR,
            "operands": [
                {
                    "type": BooleanType.AND,
                    "operands": [
                        {
                            "type": OperationType.IN,
                            "allowed_values": ["GAD", "PPI"],
                            "value": {
                                "type": DataType.COLUMN,
                                "name": "matching_specificity"
                            }
                        },
                        {
                            "type": OperationType.LESS_THAN,
                            "threshold": 0.001,
                            "value": {
                                "type": DataType.COLUMN,
                                "name": "p_val"
                            }
                        },
                    ]
                },
                {
                    "type": BooleanType.AND,
                    "operands": [
                        {
                            "type": OperationType.IN,
                            "allowed_values": ["yes"],
                            "value": {
                                "type": DataType.COLUMN,
                                "name": "a"
                            }
                        },
                        {
                            "type": OperationType.GREATER_THAN,
                            "threshold": 0.5,
                            "value": {
                                "type": DataType.COLUMN,
                                "name": "odds_ratio"
                            }
                        },
                    ]
                },
            ]
        }

        matcher = CriteriaMatcher()
        result = matcher.match(filter_params, df)
        self.assertTrue(np.array_equal(result, np.array([True, False, True, False, False, True, False])))

    def test_match_simple(self):
        df = pd.DataFrame({"matching_specificity": ["flu", "ebv", "GAD", "PPI", "GAD", "PPI", "ebv"],
                           "p_val": [0.01, 0.00001, 0.000001, 0.01, 0.01, 0.0000001, 0.1],
                           "odds_ratio": [0.51, 0.5, 0, 0, 0, 0, 0],
                           "a": ["yes", "no", "no", "no", "no", "no", "no"],
                           "b": ["no", "yes", "no", "no", "no", "no", "no"]})

        filter_params = {
            "type": OperationType.TOP_N,
            "value": {
                "type": DataType.COLUMN,
                "name": "odds_ratio",
            },
            "number": 3
        }

        matcher = CriteriaMatcher()
        result = matcher.match(filter_params, df)
        self.assertTrue(np.array_equal(result, np.array([True, True, False, False, False, False, True])))

    def test_match_none(self):
        df = pd.DataFrame({"matching_specificity": ["flu", "ebv", "GAD", "PPI", "GAD", "PPI", "ebv"],
                           "p_val": [0.01, 0.00001, 0.000001, 0.01, 0.01, 0.0000001, 0.1],
                           "odds_ratio": [0.51, 0, 0, 0, 0, 0, 0],
                           "a": ["yes", "no", "no", "no", "no", "no", "no"],
                           "b": ["no", "yes", "no", "no", "no", "no", "no"]})

        filter_params = {
            "type": OperationType.IN,
            "value": {
                "type": DataType.COLUMN,
                "name": "matching_specificity",
            },
            "allowed_values": ["GAD", "PPI"]
        }

        matcher = CriteriaMatcher()
        result = matcher.match(filter_params, df)
        self.assertTrue(np.array_equal(result, np.array([False, False, True, True, True, True, False])))
