import os
import shutil

import pandas as pd
import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.dsl.instruction_parsers.SplitDatasetParser import SplitDatasetParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.Constants import Constants
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
    }, 'type': 'SplitDataset'}, symbol_table, path)

    state = instruction.run(path / 'result')

    assert (state.test_data_paths[0] / 'subset_d1_test.tsv').is_file()
    assert (state.test_data_paths[0] / 'subset_d1_test.yaml').is_file()

    assert (state.train_data_paths[0] / 'subset_d1_train.tsv').is_file()
    assert (state.train_data_paths[0] / 'subset_d1_train.yaml').is_file()

    d1_test = SequenceDataset.build(state.test_data_paths[0] / 'subset_d1_test.tsv',
                                    state.test_data_paths[0] / 'subset_d1_test.yaml')

    assert d1_test.get_example_count() == 2

    d1_train = SequenceDataset.build(state.train_data_paths[0] / 'subset_d1_train.tsv',
                                    state.train_data_paths[0] / 'subset_d1_train.yaml')

    assert d1_train.get_example_count() == 2

    shutil.rmtree(path)


def test_split_dataset_instruction_group_k_fold():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'split_dataset_group_k_fold')

    # study s1 is always in training; the receptor reported by s2 and s3 is used only in the split where neither
    # of the two is held out
    studies = ['s1', 's1', 's1', 's2', 's2', 's3', 's3', 's4', 's4', 's2;s3']
    pd.DataFrame({'sequence_id': [f'seq{i}' for i in range(len(studies))],
                  'junction_aa': ['CASSLGQAYEQYF', 'CASSPGTEAFF', 'CASSQDRGNTIYF', 'CASSLAGGPYEQYF',
                                  'CASSIRSSYEQYF', 'CASSLEGQGYTF', 'CASSFGTDTQYF', 'CASSPRGNEKLFF',
                                  'CASSYSGNTIYF', 'CASSLDRGQPQHF'],
                  'v_call': ['TRBV7-9*01'] * len(studies), 'j_call': ['TRBJ2-7*01'] * len(studies),
                  'locus': ['TRB'] * len(studies), 'study_id': studies}).to_csv(path / 'data.tsv', sep='\t',
                                                                                    index=False)

    specs = {
        'definitions': {'datasets': {'d1': {'format': 'AIRR', 'params': {
            'path': str(path / 'data.tsv'), 'is_repertoire': False, 'paired': False,
            'region_type': 'IMGT_CDR3', 'label_columns': ['study_id']}}}},
        'instructions': {'split_by_study': {
            'type': 'SplitDataset', 'dataset': 'd1',
            'split_config': {'split_strategy': 'group_k_fold', 'split_count': 3,
                             'group_k_fold_config': {'group_label': 'study_id', 'delimiter': ';',
                                                     'train_only_groups': ['s1']}}}},
        'output': {'format': 'HTML'}
    }

    with open(path / 'specs.yaml', 'w') as file:
        yaml.dump(specs, file)

    ImmuneMLApp(specification_path=path / 'specs.yaml', result_path=path / 'result').run()

    held_out = []
    for split in range(1, 4):
        split_path = path / f'result/split_by_study/split_{split}'
        train = pd.read_csv(next((split_path / 'train').glob('*.tsv')), sep='\t')
        test = pd.read_csv(next((split_path / 'test').glob('*.tsv')), sep='\t')

        assert 's1' in set(train['study_id'])
        assert len(set(train['study_id']) & set(test['study_id'])) == 0
        assert (split_path / 'group_assignment.csv').is_file()
        held_out.extend(set(test['study_id']))

    assert sorted(held_out) == ['s2', 's3', 's4']
    assert (path / 'result/HTML_output/SplitDataset_split_by_study.html').is_file()

    shutil.rmtree(path)
