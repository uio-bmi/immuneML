from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.util.Logger import log
from immuneML.util.ReflectionHandler import ReflectionHandler


class DimensionalityReductionParser:
    @staticmethod
    def parse_dim_reductions(dim_reductions: dict, symbol_table: SymbolTable):
        if dim_reductions is None or len(dim_reductions) == 0:
            dim_reductions = {}

        for dim_id in dim_reductions.keys():
            symbol_table, dim_reductions[dim_id] = DimensionalityReductionParser._parse_dim_reduction(dim_id, dim_reductions[dim_id], symbol_table)

        return symbol_table, dim_reductions

    @staticmethod
    @log
    def _parse_dim_reduction(key: str, params: dict, symbol_table: SymbolTable):
        valid_values = ReflectionHandler.all_nonabstract_subclass_basic_names(UnsupervisedMLMethod, "", "ml_methods/")
        dim_reduction_object, params = ObjectParser.parse_object(params, valid_values, "", "ml_methods/", "DimensionalityReductionParser", key, builder=True,
                                                          return_params_dict=True)
        dim_reduction_object.name = key
        symbol_table.add(key, SymbolType.DIMENSIONAL, dim_reduction_object)
        return symbol_table, params
