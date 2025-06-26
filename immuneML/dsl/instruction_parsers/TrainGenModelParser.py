import inspect
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.hyperparameter_optimization.config.ManualSplitConfig import ManualSplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import TrainGenModelInstruction


class TrainGenModelParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> TrainGenModelInstruction:
        valid_keys = [k for k in inspect.signature(TrainGenModelInstruction.__init__).parameters.keys()
                      if k not in ['result_path', 'name', 'self']] + ['type', 'method']

        ParameterValidator.assert_all_in_valid_list(list(instruction.keys()), valid_keys, TrainGenModelParser.__name__, key)

        dataset = symbol_table.get(instruction['dataset'])

        assert ("method" in instruction) ^ ("methods" in instruction), \
            f"{TrainGenModelParser.__name__}: either 'method' or 'methods' key must be specified in the instruction."

        if 'method' in instruction:
            methods = [symbol_table.get(instruction['method'])]
        else:
            methods = [symbol_table.get(method_id) for method_id in instruction['methods']]

        assert type(dataset).__name__ in ['SequenceDataset', 'ReceptorDataset'], \
            (f'{TrainGenModelParser.__name__}: TrainGenModel instruction can for now be used only with '
             f'Sequence/Receptor datasets; Repertoire datasets are not supported.')

        for method in methods:
            assert method.__class__.__name__ not in ['ExperimentalImport', 'OLGA'], \
                (f"{TrainGenModelParser.__name__}: ExperimentalImport and OLGA cannot be used with TrainGenModel "
                 f"instruction. Please specify some of the other generative models.")

        ParameterValidator.assert_type_and_value(instruction['gen_examples_count'], int, TrainGenModelParser.__name__,
                                                 'gen_examples_count', 0)
        ParameterValidator.assert_type_and_value(instruction['number_of_processes'], int, TrainGenModelParser.__name__,
                                                 'number_of_processes', 1)
        ParameterValidator.assert_type_and_value(float(instruction['training_percentage']), float,
                                                 TrainGenModelParser.__name__,
                                                 'training_percentage', 0, 1)
        ParameterValidator.assert_type_and_value(instruction['export_generated_dataset'], bool,
                                                 TrainGenModelParser.__name__, 'export_generated_dataset')
        ParameterValidator.assert_type_and_value(instruction['export_combined_dataset'], bool,
                                                 TrainGenModelParser.__name__, 'export_combined_dataset')
        ParameterValidator.assert_type_and_value(instruction['split_strategy'], str, TrainGenModelParser.__name__, 'split_strategy')
        ParameterValidator.assert_type_and_value(instruction['split_config'], dict, TrainGenModelParser.__name__, 'split_config', nullable=True)

        assert instruction['export_generated_dataset'] or instruction['export_combined_dataset'], \
            (f"{TrainGenModelParser.__name__}: 'export_generated_dataset' and 'export_combined_dataset' are both set "
             f"to False. At least one of these must be True. ")

        valid_report_ids = symbol_table.get_keys_by_type(SymbolType.REPORT)
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_report_ids, TrainGenModelParser.__name__, 'reports')

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]

        if instruction['split_strategy'].upper() == SplitType.RANDOM.name:
            split_strategy = SplitType.RANDOM
            split_config = None
        elif instruction['split_strategy'].upper() == SplitType.MANUAL.name:
            split_strategy = SplitType.MANUAL
            split_config = ManualSplitConfig(**instruction['split_config'])
        else:
            raise ValueError(f"{TrainGenModelParser.__name__}: Unsupported split strategy '{instruction['split_strategy']}'.")

        return TrainGenModelInstruction(**{**{k: v for k, v in instruction.items() if k not in ['type', 'method']},
                                           **{'dataset': dataset, 'methods': methods, 'name': key, 'reports': reports,
                                              'split_strategy': split_strategy, 'split_config': split_config}})
