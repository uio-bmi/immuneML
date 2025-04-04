from random import randrange

import numpy as np
import pandas as pd

from immuneML.analysis.criteria_matches.BooleanType import BooleanType
from immuneML.analysis.criteria_matches.OperationType import OperationType


class CriteriaMatcher:
    """
    Takes a data frame (for example, repertoire or feature annotations) and criteria and allowed values
    as input and returns a list of boolean values indicating a match or not for each row

    """

    def match(self, criteria, data):
        return CriteriaMatcher.parse_criteria(criteria, data)

    @staticmethod
    def evaluate_in(data: pd.Series, criteria: dict):
        result = data.isin(criteria["values"])
        return result.values

    @staticmethod
    def evaluate_not_in(data: pd.Series, criteria: dict):
        result = ~data.isin(criteria["values"])
        return result.values

    @staticmethod
    def evaluate_not_na(data: pd.Series, criteria: dict):
        result = data.notna()
        return result.values

    @staticmethod
    def evaluate_greater_than(data: pd.Series, criteria: dict):
        result = data > criteria["threshold"]
        return result.values

    @staticmethod
    def evaluate_less_than(data: pd.Series, criteria: dict):
        result = data < criteria["threshold"]
        return result.values

    @staticmethod
    def evaluate_top_n(data: pd.Series, criteria: dict):
        top_n = data.values.argsort()[(-1 * criteria["number"]):][::-1]
        result = [i in top_n for i in range(data.size)]
        return np.array(result)

    @staticmethod
    def evaluate_random_n(data: pd.Series, criteria: dict):
        random_n = [randrange(0, data.size) for i in range(criteria["number"])]
        result = [i in random_n for i in range(data.size)]
        return np.array(result)

    @staticmethod
    def evaluate_and(booleans: list):
        result = np.logical_and.reduce(booleans)
        return result

    @staticmethod
    def evaluate_or(booleans: list):
        result = np.logical_or.reduce(booleans)
        return result

    @staticmethod
    def evaluate_column(data: pd.Series, name: str) -> pd.Series:
        return data[name]

    @staticmethod
    def parse_criteria(criteria, data):
        if criteria["type"] in OperationType:
            operation = getattr(CriteriaMatcher, "evaluate_" + criteria["type"].name.lower())
            return operation(data[criteria['column']], criteria)
        elif criteria["type"] in BooleanType:
            operation = getattr(CriteriaMatcher, "evaluate_" + criteria["type"].name.lower())
            booleans = []
            for operand in criteria["operands"]:
                booleans.append(CriteriaMatcher.parse_criteria(operand, data))
            return operation(booleans)
