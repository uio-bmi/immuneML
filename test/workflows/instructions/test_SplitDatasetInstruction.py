import shutil

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.dsl.instruction_parsers.SplitDatasetParser import SplitDatasetParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_split_dataset_instruction():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'split_dataset_instruction')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(4, {3: 1}, {'label1': {1: 0.5, 2: 0.5}}, path / 'dataset', name='d1')

    symbol_table = SymbolTable()
    symbol_table.add('d1', SymbolType.DATASET, dataset)

    instruction = SplitDatasetParser().parse('inst1', {'dataset': 'd1', 'split_config': {
        'split_strategy': 'RANDOM',
        'split_count': 1,
        'training_percentage': 0.5
    }}, symbol_table, path)

    state = instruction.run(path / 'result')

    assert (state.test_data_path / 'subset_d1_test.tsv').is_file()
    assert (state.test_data_path / 'subset_d1_test.yaml').is_file()

    assert (state.train_data_path / 'subset_d1_train.tsv').is_file()
    assert (state.train_data_path / 'subset_d1_train.yaml').is_file()

    d1_test = SequenceDataset.build(state.test_data_path / 'subset_d1_test.tsv',
                                    state.test_data_path / 'subset_d1_test.yaml')

    assert d1_test.get_example_count() == 2

    d1_train = SequenceDataset.build(state.train_data_path / 'subset_d1_train.tsv',
                                    state.train_data_path / 'subset_d1_train.yaml')

    assert d1_train.get_example_count() == 2

    shutil.rmtree(path)