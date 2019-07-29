from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


class PreprocessingParser:

    keyword = "preprocessing_sequences"

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        if PreprocessingParser.keyword in workflow_specification:
            for key in workflow_specification[PreprocessingParser.keyword]:
                symbol_table = PreprocessingParser._parse_sequence(
                    workflow_specification[PreprocessingParser.keyword][key],
                    key,
                    symbol_table)
        else:
            workflow_specification[PreprocessingParser.keyword] = {}

        return symbol_table, workflow_specification[PreprocessingParser.keyword]

    @staticmethod
    def _parse_sequence(preproc_sequence: list, key: str, symbol_table: SymbolTable) -> SymbolTable:

        sequence = []

        for item in preproc_sequence:
            for step_key in item:
                cls = ReflectionHandler.get_class_by_name(item[step_key]["type"], "preprocessing/")
                obj = cls(**item[step_key]["params"])
                sequence.append(obj)

        symbol_table.add(key, SymbolType.PREPROCESSING, sequence)
        return symbol_table
