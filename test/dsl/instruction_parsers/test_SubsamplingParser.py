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
                                   'subsampled_dataset_sizes': [10, 20], 'subsampled_repertoire_size': None},
                                  symbol_table)

        shutil.rmtree(path)
