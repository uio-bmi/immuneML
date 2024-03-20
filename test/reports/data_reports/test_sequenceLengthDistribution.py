import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceLengthDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_get_normalized_sequence_lengths(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_rep")

        rep1 = Repertoire.build_from_sequence_objects(sequence_objects=[ReceptorSequence(sequence_aa="AAA", sequence_id="1"),
                                                                        ReceptorSequence(sequence_aa="AAAA", sequence_id="2"),
                                                                        ReceptorSequence(sequence_aa="AAAAA", sequence_id="3"),
                                                                        ReceptorSequence(sequence_aa="AAA", sequence_id="4")],
                                                      path=path, metadata={})
        rep2 = Repertoire.build_from_sequence_objects(sequence_objects=[ReceptorSequence(sequence_aa="AAA", sequence_id="5"),
                                                                        ReceptorSequence(sequence_aa="AAAA", sequence_id="6"),
                                                                        ReceptorSequence(sequence_aa="AAAA", sequence_id="7"),
                                                                        ReceptorSequence(sequence_aa="AAA", sequence_id="8")],
                                                      path=path, metadata={})

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        sld = SequenceLengthDistribution.build_object(dataset=dataset, sequence_type='amino_acid', result_path=path)

        self.assertTrue(sld.check_prerequisites())
        result = sld._generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)

    def test_sequence_lengths_seq_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_seq")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {4: 0.33, 5: 0.33, 7: 0.33}, {}, path / 'dataset')

        sld = SequenceLengthDistribution(dataset, 1, path, sequence_type=SequenceType.AMINO_ACID)

        self.assertTrue(sld.check_prerequisites())
        result = sld._generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)
