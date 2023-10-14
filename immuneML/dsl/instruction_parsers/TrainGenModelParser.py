import inspect
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import TrainGenModelInstruction


class TrainGenModelParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> TrainGenModelInstruction:
        valid_keys = [k for k in inspect.signature(TrainGenModelInstruction.__init__).parameters.keys()
                      if k not in ['result_path', 'name', 'self']] + ['type']

        ParameterValidator.assert_keys(instruction.keys(), valid_keys, TrainGenModelParser.__name__, key)

        dataset = symbol_table.get(instruction['dataset'])
        model = symbol_table.get(instruction['method'])

        ParameterValidator.assert_type_and_value(instruction['gen_examples_count'], int, TrainGenModelParser.__name__,
                                                 'gen_examples_count', 0)
        ParameterValidator.assert_type_and_value(instruction['number_of_processes'], int, TrainGenModelParser.__name__,
                                                 'number_of_processes', 1)

        valid_report_ids = symbol_table.get_keys_by_type(SymbolType.REPORT)
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_report_ids, TrainGenModelParser.__name__, 'reports')

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]

        return TrainGenModelInstruction(**{**{k: v for k, v in instruction.items() if k != 'type'},
                                           **{'dataset': dataset, 'method': model, 'name': key, 'reports': reports}})
