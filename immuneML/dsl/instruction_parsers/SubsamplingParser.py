from pathlib import Path

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.subsampling.SubsamplingInstruction import SubsamplingInstruction


class SubsamplingParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> SubsamplingInstruction:
        valid_keys = ["type", "dataset", "subsampled_dataset_sizes", "subsampled_repertoire_size"]
        ParameterValidator.assert_keys(instruction.keys(), valid_keys, SubsamplingParser.__name__, key)

        dataset_keys = symbol_table.get_keys_by_type(SymbolType.DATASET)
        ParameterValidator.assert_in_valid_list(instruction['dataset'], dataset_keys, SubsamplingParser.__name__, f'{key}/dataset')

        dataset = symbol_table.get(instruction['dataset'])
        ParameterValidator.assert_type_and_value(instruction['subsampled_dataset_sizes'], list, SubsamplingParser.__name__, f'{key}/subsampled_dataset_sizes')
        ParameterValidator.assert_all_type_and_value(instruction['subsampled_dataset_sizes'], int, SubsamplingParser.__name__,
                                                     f'{key}/subsampled_dataset_sizes', 1, dataset.get_example_count())

        ParameterValidator.assert_type_and_value(instruction['subsampled_repertoire_size'], int,
                                                 SubsamplingParser.__name__, f'{key}/subsampled_repertoire_size',
                                                 nullable=True)

        return SubsamplingInstruction(dataset=dataset,
                                      subsampled_repertoire_size=instruction['subsampled_repertoire_size'],
                                      subsampled_dataset_sizes=instruction['subsampled_dataset_sizes'],
                                      name=key)
