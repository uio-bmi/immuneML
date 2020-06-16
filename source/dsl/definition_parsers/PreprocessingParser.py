from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.logging.Logger import log
from source.util.ReflectionHandler import ReflectionHandler


class PreprocessingParser:

    keyword = "preprocessing_sequences"

    @staticmethod
    def parse(specs: dict, symbol_table: SymbolTable):
        for key in specs:
            symbol_table = PreprocessingParser._parse_sequence(key, specs[key], symbol_table)

        return symbol_table, specs

    @staticmethod
    @log
    def _parse_sequence(key: str, preproc_sequence: list, symbol_table: SymbolTable) -> SymbolTable:

        sequence = []

        for item in preproc_sequence:
            for step_key, step in item.items():
                if isinstance(step, str):
                    class_name = step
                    params = {}
                else:
                    class_name = list(item[step_key].keys())[0]
                    params = step[class_name]
                cls = ReflectionHandler.get_class_by_name(class_name, "preprocessing/")
                obj = cls(**params)
                sequence.append(obj)

        symbol_table.add(key, SymbolType.PREPROCESSING, sequence)
        return symbol_table
