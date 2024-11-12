import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceLengthDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_sequence_lengths_rep_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_rep")

        rep1 = Repertoire.build_from_sequences(sequences=[ReceptorSequence(sequence_aa="AAA", sequence_id="1"),
                                                          ReceptorSequence(sequence_aa="AAAA", sequence_id="2"),
                                                          ReceptorSequence(sequence_aa="AAAAA", sequence_id="3"),
                                                          ReceptorSequence(sequence_aa="AAA", sequence_id="4")],
                                               result_path=path, metadata={})
        rep2 = Repertoire.build_from_sequences(sequences=[ReceptorSequence(sequence_aa="AAA", sequence_id="5"),
                                                          ReceptorSequence(sequence_aa="AAAA", sequence_id="6"),
                                                          ReceptorSequence(sequence_aa="AAAA", sequence_id="7"),
                                                          ReceptorSequence(sequence_aa="AAA", sequence_id="8")],
                                               result_path=path, metadata={})

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        sld = SequenceLengthDistribution.build_object(dataset=dataset, sequence_type='amino_acid', result_path=path,
                                                      region_type='IMGT_CDR3')

        self.assertTrue(sld.check_prerequisites())
        result = sld._generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)

    def test_sequence_lengths_seq_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_seq")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {4: 0.33, 5: 0.33, 7: 0.33}, {},
                                                                   path / 'dataset')

        sld = SequenceLengthDistribution(dataset, 1, path, sequence_type=SequenceType.AMINO_ACID,
                                         region_type=RegionType.IMGT_CDR3)

        self.assertTrue(sld.check_prerequisites())
        result = sld._generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)

    def test_sequence_lengths_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_rec")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=50,
                                                                   chain_1_length_probabilities={4: 0.33, 5: 0.33,
                                                                                                 7: 0.33},
                                                                   chain_2_length_probabilities={7: 0.33, 8: 0.33,
                                                                                                 9: 0.33},
                                                                   labels={}, path=path / 'dataset')

        sld = SequenceLengthDistribution(dataset, 1, path, sequence_type=SequenceType.AMINO_ACID,
                                         region_type=RegionType.IMGT_CDR3)

        result = sld.generate_report()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)
