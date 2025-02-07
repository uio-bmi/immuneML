from immuneML.analysis.criteria_matches.BooleanType import BooleanType
from immuneML.analysis.criteria_matches.OperationType import OperationType


class CriteriaTypeInstantiator:

    @staticmethod
    def instantiate(criteria):
        if criteria["type"].upper() in OperationType._member_names_:
            return {**criteria, "type": OperationType[criteria["type"].upper()]}
        elif criteria["type"].upper() in BooleanType._member_names_:
            operands = []
            for operand in criteria["operands"]:
                operands.append(CriteriaTypeInstantiator.instantiate(operand))
            return {**criteria, "operands": operands, "type": BooleanType[criteria["type"].upper()]}
