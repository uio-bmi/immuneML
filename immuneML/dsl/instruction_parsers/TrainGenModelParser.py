import inspect
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import TrainGenModelInstruction


class TrainGenModelParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> TrainGenModelInstruction:
        valid_keys = [k for k in inspect.signature(TrainGenModelInstruction.__init__).parameters.keys()
                      if k not in ['result_path', 'name', 'self']] + ['type']
        ParameterValidator.assert_keys(instruction.keys(), valid_keys, TrainGenModelParser.__name__, key)

        dataset = symbol_table.get(instruction['dataset'])
        model = symbol_table.get(instruction['model'])

        ParameterValidator.assert_type_and_value(instruction['gen_sequence_count'], int, TrainGenModelParser.__name__,
                                                 'gen_sequence_count', 0)
        ParameterValidator.assert_type_and_value(instruction['number_of_processes'], int, TrainGenModelParser.__name__,
                                                 'number_of_processes', 1)

        return TrainGenModelInstruction(**{**{k: v for k, v in instruction.items() if k != 'type'},
                                           **{'dataset': dataset, 'model': model, 'name': key}})
