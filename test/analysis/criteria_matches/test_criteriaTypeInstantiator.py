import os
from unittest import TestCase

from immuneML.analysis.criteria_matches.BooleanType import BooleanType
from immuneML.analysis.criteria_matches.CriteriaTypeInstantiator import CriteriaTypeInstantiator
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants


class TestCriteriaTypeInstantiator(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_instantiate(self):

        filter_params = {
            "type": "or",
            "operands": [
                {
                    "type": "and",
                    "operands": [
                        {
                            "type": "in",
                            "allowed_values": ["GAD", "PPI"],
                            "value": {
                                "type": "column",
                                "name": "matching_specificity"
                            }
                        },
                        {
                            "type": "less_than",
                            "threshold": 0.001,
                            "value": {
                                "type": "column",
                                "name": "p_val"
                            }
                        },
                    ]
                },
                {
                    "type": "and",
                    "operands": [
                        {
                            "type": "in",
                            "allowed_values": ["yes"],
                            "value": {
                                "type": "COLUMN",
                                "name": "a"
                            }
                        },
                        {
                            "type": "greater_than",
                            "threshold": 0.5,
                            "value": {
                                "type": "column",
                                "name": "odds_ratio"
                            }
                        },
                    ]
                },
            ]
        }

        result = CriteriaTypeInstantiator.instantiate(filter_params)

        self.assertEqual(result["operands"][0]["type"], BooleanType.AND)
