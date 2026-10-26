import os
import shutil

import pandas as pd
import pytest
import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
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


def test_split_dataset_instruction_similarity_and_sequence_overlap():
    compairr_path = next((p for p in EnvironmentSettings.compairr_paths if p.exists()), None)
    assert compairr_path is not None, "CompAIRR binary not found; build it to run this test"

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'split_dataset_similarity')

    # four families of sequences within one substitution of each other, plus an exact duplicate in the first one
    junctions = ['CASSLGQAYEQYF', 'CASSLGQAYEQYF', 'CASSLGQAYEHYF', 'CASSLGQGYEHYF',
                 'CASSPGTEAFF', 'CASSPGTEAYF',
                 'CASRQDRGNTIYF', 'CASRQDRGNTLYF',
                 'CAWSVGDTQYF', 'CAWSVGDTHYF']
    pd.DataFrame({'sequence_id': [f'seq{i}' for i in range(len(junctions))], 'junction_aa': junctions,
                  'v_call': ['TRBV7-9*01'] * len(junctions), 'j_call': ['TRBJ2-7*01'] * len(junctions),
                  'locus': ['TRB'] * len(junctions)}).to_csv(path / 'data.tsv', sep='\t', index=False)

    overlap_config = {'max_edit_distance': 2, 'compairr_path': str(compairr_path)}
    specs = {
        'definitions': {'datasets': {'d1': {'format': 'AIRR', 'params': {
            'path': str(path / 'data.tsv'), 'is_repertoire': False, 'paired': False, 'region_type': 'IMGT_CDR3'}}}},
        'instructions': {
            'split_by_similarity': {
                'type': 'SplitDataset', 'dataset': 'd1',
                'split_config': {'split_strategy': 'group_k_fold', 'split_count': 2,
                                 'group_k_fold_config': {'similarity': {'max_edit_distance': 1,
                                                                        'compairr_path': str(compairr_path)}}},
                'sequence_overlap': overlap_config},
            'split_random': {
                'type': 'SplitDataset', 'dataset': 'd1',
                'split_config': {'split_strategy': 'random', 'split_count': 1, 'training_percentage': 0.5},
                'sequence_overlap': overlap_config}},
        'output': {'format': 'HTML'}
    }

    with open(path / 'specs.yaml', 'w') as file:
        yaml.dump(specs, file)

    ImmuneMLApp(specification_path=path / 'specs.yaml', result_path=path / 'result').run()

    for split in range(1, 3):
        split_path = path / f'result/split_by_similarity/split_{split}'
        train = pd.read_csv(next((split_path / 'train').glob('*.tsv')), sep='\t')
        test = pd.read_csv(next((split_path / 'test').glob('*.tsv')), sep='\t')
        assert len(set(train['junction_aa']) & set(test['junction_aa'])) == 0

        # by construction, no test sequence is within one substitution of a training sequence
        overlap = pd.read_csv(split_path / 'sequence_overlap.csv', dtype={'distance_to_training_set': str})
        assert list(overlap['distance_to_training_set']) == ['0', '1', '2', 'more than 2']
        assert overlap['test_sequences'].iloc[0] == 0 and overlap['test_sequences'].iloc[1] == 0
        assert overlap['test_sequences'].sum() == test.shape[0]

    overlap = pd.read_csv(path / 'result/split_random/sequence_overlap.csv')
    assert overlap['test_sequences'].sum() == 5
    assert abs(overlap['fraction_of_test_set'].sum() - 1) < 1e-9

    assert (path / 'result/HTML_output/SplitDataset_split_by_similarity.html').is_file()
    assert (path / 'result/HTML_output/SplitDataset_split_random.html').is_file()

    shutil.rmtree(path)


def test_split_dataset_instruction_receptors_by_beta_chain():
    compairr_path = next((p for p in EnvironmentSettings.compairr_paths if p.exists()), None)
    assert compairr_path is not None, "CompAIRR binary not found; build it to run this test"

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'split_dataset_receptors')

    # three families of beta chains within one substitution, each paired with unrelated alpha chains
    betas = ['CASSLGQAYEQYF', 'CASSLGQAYEHYF', 'CASSLGQGYEHYF', 'CASSPGTEAFF', 'CASSPGTEAYF', 'CAWSVGDTQYF',
             'CAWSVGDTHYF']
    alphas = ['CAVRDSNYQLIW', 'CILRDGQKLLF', 'CAMREGRYGGSQGNLIF', 'CAASGGSYIPTF', 'CVVNDYKLSF', 'CALSEAGNNRKLIW',
              'CAGPTGGSNYKLTF']
    receptors = [Receptor(chain_1=ReceptorSequence(sequence_id=f'a{i}', sequence_aa=alpha, locus='TRA'),
                          chain_2=ReceptorSequence(sequence_id=f'b{i}', sequence_aa=beta, locus='TRB'),
                          chain_pair=ChainPair.TRA_TRB, receptor_id=f'r{i}', cell_id=f'c{i}')
                 for i, (alpha, beta) in enumerate(zip(alphas, betas))]
    dataset = ReceptorDataset.build_from_objects(receptors, PathBuilder.build(path / 'dataset'), name='d1')

    symbol_table = SymbolTable()
    symbol_table.add('d1', SymbolType.DATASET, dataset)
    similarity = {'max_edit_distance': 1, 'chains': ['TRB'], 'compairr_path': str(compairr_path)}

    instruction = SplitDatasetParser().parse('inst1', {
        'dataset': 'd1', 'type': 'SplitDataset',
        'split_config': {'split_strategy': 'group_k_fold', 'split_count': 2,
                         'group_k_fold_config': {'similarity': similarity}},
        'sequence_overlap': {**similarity, 'max_edit_distance': 2}}, symbol_table, path)
    state = instruction.run(path / 'result')

    for split in range(2):
        overlap = pd.read_csv(state.sequence_overlap_paths[split], dtype={'distance_to_training_set': str})
        # the beta chains are split by family, so no test receptor's beta is within one substitution of training
        assert overlap['test_sequences'].iloc[0] == 0 and overlap['test_sequences'].iloc[1] == 0
        assert overlap['test_sequences'].sum() > 0

    shutil.rmtree(path)


def test_split_dataset_parser_rejects_similarity_with_train_only_groups():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'split_dataset_similarity_parser')
    dataset = RandomDatasetGenerator.generate_sequence_dataset(4, {3: 1}, {'label1': {1: 0.5, 2: 0.5}},
                                                               path / 'dataset', name='d1')
    symbol_table = SymbolTable()
    symbol_table.add('d1', SymbolType.DATASET, dataset)

    with pytest.raises(AssertionError):
        SplitDatasetParser().parse('inst1', {'dataset': 'd1', 'type': 'SplitDataset', 'split_config': {
            'split_strategy': 'group_k_fold', 'split_count': 2,
            'group_k_fold_config': {'similarity': {'max_edit_distance': 1}, 'train_only_groups': ['a']}}},
                                   symbol_table, path)

    shutil.rmtree(path)
