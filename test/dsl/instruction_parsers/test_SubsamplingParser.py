import shutil
from unittest import TestCase

from immuneML.dsl.instruction_parsers.SubsamplingParser import SubsamplingParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSubsamplingParser(TestCase):
    def test_parse(self):

        path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/subsampling_parser/')
        dataset = RandomDatasetGenerator.generate_receptor_dataset(30, {3: 1}, {2: 1}, {}, path)

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)

        SubsamplingParser().parse('inst1',
                                  {'dataset': 'd1', 'type': 'Subsampling',
                                   'subsampled_dataset_sizes': [10, 20], 'subsampled_repertoire_size': None,
                                   'label': None, 'subsampled_class_distributions': None},
                                  symbol_table)

        shutil.rmtree(path)

    def test_parse_class_balanced(self):
        path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/subsampling_parser_class_balanced/')
        dataset = RandomDatasetGenerator.generate_receptor_dataset(100, {3: 1}, {2: 1}, {"epitope": {"A": 0.5, "B": 0.5}}, path)

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)

        instruction = SubsamplingParser().parse('inst1',
                                                 {'dataset': 'd1', 'type': 'Subsampling',
                                                  'subsampled_dataset_sizes': [20, 20],
                                                  'subsampled_repertoire_size': None,
                                                  'label': 'epitope',
                                                  # single fraction is enough for a binary label, the other class is inferred
                                                  'subsampled_class_distributions': [{"A": 0.1}, {"A": 0.5}]},
                                                 symbol_table)

        self.assertEqual("epitope", instruction.state.label.name)
        self.assertEqual([{"A": 0.1, "B": 0.9}, {"A": 0.5, "B": 0.5}], instruction.state.subsampled_class_distributions)

        shutil.rmtree(path)

    def test_parse_class_balanced_label_and_distribution_must_both_be_set(self):
        path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/subsampling_parser_class_balanced_missing/')
        dataset = RandomDatasetGenerator.generate_receptor_dataset(30, {3: 1}, {2: 1}, {"epitope": {"A": 0.5, "B": 0.5}}, path)

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)

        with self.assertRaises(AssertionError):
            SubsamplingParser().parse('inst1',
                                      {'dataset': 'd1', 'type': 'Subsampling',
                                       'subsampled_dataset_sizes': [10, 20], 'subsampled_repertoire_size': None,
                                       'label': None, 'subsampled_class_distributions': [{"A": 0.5}, {"A": 0.5}]},
                                      symbol_table)

        shutil.rmtree(path)

    def test_parse_class_balanced_wrong_number_of_distributions(self):
        path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/subsampling_parser_class_balanced_length/')
        dataset = RandomDatasetGenerator.generate_receptor_dataset(30, {3: 1}, {2: 1}, {"epitope": {"A": 0.5, "B": 0.5}}, path)

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)

        with self.assertRaises(AssertionError):
            SubsamplingParser().parse('inst1',
                                      {'dataset': 'd1', 'type': 'Subsampling',
                                       'subsampled_dataset_sizes': [10, 20], 'subsampled_repertoire_size': None,
                                       'label': 'epitope', 'subsampled_class_distributions': [{"A": 0.5}]},
                                      symbol_table)

        shutil.rmtree(path)

    def test_parse_class_balanced_single_fraction_requires_binary_label(self):
        path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/subsampling_parser_class_balanced_multiclass/')
        dataset = RandomDatasetGenerator.generate_receptor_dataset(60, {3: 1}, {2: 1},
                                                                    {"epitope": {"A": 0.34, "B": 0.33, "C": 0.33}}, path)

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)

        with self.assertRaises(AssertionError):
            SubsamplingParser().parse('inst1',
                                      {'dataset': 'd1', 'type': 'Subsampling',
                                       'subsampled_dataset_sizes': [10], 'subsampled_repertoire_size': None,
                                       'label': 'epitope', 'subsampled_class_distributions': [{"A": 0.5}]},
                                      symbol_table)

        shutil.rmtree(path)

    def test_parse_class_balanced_infeasible(self):
        path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/subsampling_parser_class_balanced_infeasible/')
        dataset = RandomDatasetGenerator.generate_receptor_dataset(30, {3: 1}, {2: 1}, {"epitope": {"A": 0.5, "B": 0.5}}, path)

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)

        # requesting all 40 examples to be class 'A' is infeasible since the dataset only has 30 examples in total
        with self.assertRaises(AssertionError):
            SubsamplingParser().parse('inst1',
                                      {'dataset': 'd1', 'type': 'Subsampling',
                                       'subsampled_dataset_sizes': [30], 'subsampled_repertoire_size': None,
                                       'label': 'epitope', 'subsampled_class_distributions': [{"A": 1.0}]},
                                      symbol_table)

        shutil.rmtree(path)
