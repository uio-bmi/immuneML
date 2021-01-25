from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.Logger import log
from immuneML.util.ReflectionHandler import ReflectionHandler


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

        valid_preprocessing_classes = ReflectionHandler.all_nonabstract_subclass_basic_names(Preprocessor, "", "preprocessing/")

        for item in preproc_sequence:
            for step_key, step in item.items():
                obj, params = ObjectParser.parse_object(step, valid_preprocessing_classes, "", "preprocessing/", "PreprocessingParser",
                                                        step_key, True, True)
                step = params
                sequence.append(obj)

        symbol_table.add(key, SymbolType.PREPROCESSING, sequence)
        return symbol_table
