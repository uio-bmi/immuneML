from unittest import TestCase

from source.analysis.criteria_matches.BooleanType import BooleanType
from source.analysis.criteria_matches.CriteriaTypeInstantiator import CriteriaTypeInstantiator


class TestCriteriaTypeInstantiator(TestCase):

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
