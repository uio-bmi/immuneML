import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.util.PathBuilder import PathBuilder


class TestSequenceLengthDistribution(TestCase):
    def test_get_normalized_sequence_lengths(self):
        path = EnvironmentSettings.root_path + "test/tmp/datareports/"
        PathBuilder.build(path)

        rep1 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA"),
                                             ReceptorSequence(amino_acid_sequence="AAAA"),
                                             ReceptorSequence(amino_acid_sequence="AAAAA"),
                                             ReceptorSequence(amino_acid_sequence="AAA")])
        rep2 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA"),
                                             ReceptorSequence(amino_acid_sequence="AAAA"),
                                             ReceptorSequence(amino_acid_sequence="AAAA"),
                                             ReceptorSequence(amino_acid_sequence="AAA")])

        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        dataset = RepertoireDataset(filenames=[path + "rep1.pkl", path + "rep2.pkl"])

        sld = SequenceLengthDistribution(dataset, 1, path)
        lengths = sld.get_normalized_sequence_lengths()

        self.assertTrue(all([key in lengths.keys() for key in [3, 4, 5]]))
        self.assertEqual(0.5, lengths[3])
        self.assertEqual(0.125, lengths[5])
        self.assertEqual(0.375, lengths[4])

        shutil.rmtree(path)
