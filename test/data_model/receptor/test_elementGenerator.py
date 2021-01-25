import pickle
import shutil
from unittest import TestCase

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.BCReceptor import BCReceptor
from immuneML.data_model.receptor.ElementGenerator import ElementGenerator
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestElementGenerator(TestCase):
    def test_build_batch_generator(self):
        path = EnvironmentSettings.tmp_test_path / "element_batch_generator/"
        PathBuilder.build(path)
        receptors = [BCReceptor(identifier=str(i)) for i in range(307)]
        file_list = [path / f"batch{i}.pkl" for i in range(4)]

        for i in range(4):
            with file_list[i].open("wb") as file:
                pickle.dump(receptors[i * 100: (i+1) * 100], file)

        receptor_generator = ElementGenerator(file_list)
        generator = receptor_generator.build_batch_generator()

        counter = 0

        for batch in generator:
            for receptor in batch:
                self.assertEqual(counter, int(receptor.identifier))
                self.assertTrue(isinstance(receptor, BCReceptor))
                counter += 1

        self.assertEqual(307, counter)

        generator = receptor_generator.build_batch_generator()

        counter = 0

        for batch in generator:
            for receptor in batch:
                self.assertEqual(counter, int(receptor.identifier))
                self.assertTrue(isinstance(receptor, BCReceptor))
                counter += 1

        self.assertEqual(307, counter)

        shutil.rmtree(path)

    def test_make_subset(self):
        sequences = []
        for i in range(100):
            sequences.append(ReceptorSequence(amino_acid_sequence="AAA", identifier=str(i)))

        path = EnvironmentSettings.tmp_test_path / "element_generator_subset/"
        PathBuilder.build(path)

        for i in range(10):
            filepath = path / f"batch{i}.pkl"
            with filepath.open("wb") as file:
                sequences_to_pickle = sequences[i * 10:(i + 1) * 10]
                pickle.dump(sequences_to_pickle, file)

        d = SequenceDataset(filenames=[path / f"batch{i}.pkl" for i in range(10)], file_size=10)

        indices = [1, 20, 21, 22, 23, 24, 25, 50, 52, 60, 70, 77, 78, 90, 92]

        d2 = d.make_subset(indices, path, SequenceDataset.TRAIN)

        for batch in d2.get_batch(1000):
            for sequence in batch:
                self.assertTrue(int(sequence.identifier) in indices)

        self.assertEqual(15, d2.get_example_count())

        shutil.rmtree(path)


