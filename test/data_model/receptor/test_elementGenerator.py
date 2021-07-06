import shutil
from unittest import TestCase

import numpy as np

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
        receptors = [BCReceptor(identifier=str(i), heavy=ReceptorSequence('A'), light=ReceptorSequence('C')) for i in range(307)]
        file_list = [path / f"batch{i}.npy" for i in range(4)]

        for i in range(4):
            matrix = np.core.records.fromrecords([r.get_record() for r in receptors[i * 100: (i+1) * 100]], names=BCReceptor.get_record_names())
            np.save(str(file_list[i]), matrix, allow_pickle=False)

        receptor_generator = ElementGenerator(file_list, element_class_name=BCReceptor.__name__)
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
            filepath = path / f"batch{i}.npy"
            sequences_to_pickle = sequences[i * 10:(i + 1) * 10]
            sequence_matrix = np.core.records.fromrecords([seq.get_record() for seq in sequences_to_pickle], names=ReceptorSequence.get_record_names())
            np.save(str(filepath), sequence_matrix, allow_pickle=False)

        d = SequenceDataset(filenames=[path / f"batch{i}.npy" for i in range(10)], file_size=10)

        indices = [1, 20, 21, 22, 23, 24, 25, 50, 52, 60, 70, 77, 78, 90, 92]

        d2 = d.make_subset(indices, path, SequenceDataset.TRAIN)

        for batch in d2.get_batch(1000):
            for sequence in batch:
                self.assertTrue(int(sequence.identifier) in indices)

        self.assertEqual(15, d2.get_example_count())

        shutil.rmtree(path)


