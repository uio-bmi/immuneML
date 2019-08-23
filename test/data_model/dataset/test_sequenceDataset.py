import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceDataset(TestCase):
    def test_make_subset(self):
        sequences = []
        for i in range(100):
            sequences.append(ReceptorSequence(amino_acid_sequence="AAA", identifier=str(i)))

        path = EnvironmentSettings.tmp_test_path + "sequencedataset/"
        PathBuilder.build(path)

        for i in range(10):
            with open("{}batch{}.pkl".format(path, i), "wb") as file:
                sequences_to_pickle = sequences[i*10:(i+1)*10]
                pickle.dump(sequences_to_pickle, file)

        d = SequenceDataset(filenames=["{}batch{}.pkl".format(path, i) for i in range(10)])

        indices = [1, 20, 21, 22, 23, 24, 25, 50, 52, 60, 70, 77, 78, 90, 92]

        d2 = d.make_subset(indices, path)

        for batch in d2.get_batch(1000):
            for sequence in batch:
                self.assertTrue(int(sequence.identifier) in indices)

        self.assertEqual(15, d2.get_example_count())

        shutil.rmtree(path)
