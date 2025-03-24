import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML import Constants
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType, ChainPair
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.SequenceLengthFilter import SequenceLengthFilter
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceLengthFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rep_seq_len_filter')

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(5, sequence_count_probabilities={50: 1.},
                                                                     sequence_length_probabilities={3: 0.5, 5: 0.3, 4: 0.2},
                                                                     labels={}, path=path / 'initial_dataset')

        filter = SequenceLengthFilter.build_object(min_len=4, max_len=-1, sequence_type='amino_acid', name='test_sl_filter',
                                                   region_type=RegionType.IMGT_CDR3.name)

        processed_dataset = filter.process_dataset(dataset, path / 'processed')

        assert len(processed_dataset.repertoires) == 5

        for repertoire in processed_dataset.repertoires:
            assert np.all(repertoire.data.cdr3_aa.lengths >= 4)

        shutil.rmtree(path)

    def test_process_sequence_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'seq_len_filter_sequence')

        dataset = RandomDatasetGenerator.generate_sequence_dataset(50, length_probabilities={3: 0.5, 5: 0.3, 4: 0.2},
                                                                   labels={}, path=path / 'initial_dataset')

        filter = SequenceLengthFilter.build_object(min_len=4, max_len=-1, sequence_type='amino_acid', name='test_sl_filter',
                                                   region_type=RegionType.IMGT_CDR3.name)

        processed_dataset = filter.process_dataset(dataset, path / 'processed')

        assert np.all(processed_dataset.data.cdr3_aa.lengths >= 4)

        shutil.rmtree(path)

    def test_process_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'seq_len_filter_receptor')

        dataset = ReceptorDataset.build_from_objects([Receptor(ChainPair.TRA_TRB,
                                                               ReceptorSequence(sequence_aa="AAAAA", cell_id='1'),
                                                               ReceptorSequence(sequence_aa="CC", cell_id='1'),
                                                               cell_id='1'),
                                                      Receptor(ChainPair.TRA_TRB,
                                                               ReceptorSequence(sequence_aa="AAAAA", cell_id='2'),
                                                               ReceptorSequence(sequence_aa="CCCCC", cell_id='2'),
                                                               cell_id='2')
                                                      ], path)

        filter = SequenceLengthFilter.build_object(min_len=4, max_len=-1, sequence_type='amino_acid', name='test_sl_filter',
                                                   region_type=RegionType.IMGT_CDR3.name)

        processed_dataset = filter.process_dataset(dataset, path / 'processed')

        assert len(processed_dataset.data) == 2
        assert processed_dataset.data.cell_id.tolist() == ['2', '2']

        shutil.rmtree(path)

