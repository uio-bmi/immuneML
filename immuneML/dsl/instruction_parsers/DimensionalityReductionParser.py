import inspect
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.dimensionality_reduction.DimensionalityReductionInstruction import \
    DimensionalityReductionInstruction
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import TrainGenModelInstruction


class DimensionalityReductionParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> DimensionalityReductionInstruction:
        valid_keys = [k for k in inspect.signature(DimensionalityReductionInstruction.__init__).parameters.keys()
                      if k not in ['result_path', 'name', 'self']] + ['type']

        ParameterValidator.assert_keys(instruction.keys(), valid_keys, DimensionalityReductionInstruction.__name__, key)

        dataset = symbol_table.get(instruction['dataset'])
        model = symbol_table.get(instruction['method'])

        valid_report_ids = symbol_table.get_keys_by_type(SymbolType.REPORT)
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_report_ids, DimensionalityReductionInstruction.__name__, 'reports')

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]

        return DimensionalityReductionInstruction(**{**{k: v for k, v in instruction.items() if k != 'type'},
                                           **{'dataset': dataset, 'method': model, 'name': key, 'reports': reports}})
