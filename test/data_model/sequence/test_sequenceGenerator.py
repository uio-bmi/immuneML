import pickle
import shutil
from unittest import TestCase

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceGenerator import SequenceGenerator
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceGenerator(TestCase):
    def test_build_generator(self):
        path = EnvironmentSettings.tmp_test_path + "sequence_generator/"
        PathBuilder.build(path)
        sequences = [ReceptorSequence(identifier=str(i)) for i in range(307)]
        file_list = ["{}batch{}.pkl".format(path, i) for i in range(4)]

        for i in range(4):
            with open(file_list[i], "wb") as file:
                pickle.dump(sequences[i * 100: (i+1) * 100], file)

        sequence_generator = SequenceGenerator(file_list)
        generator = sequence_generator.build_generator(41)

        counter = 0

        for batch in generator:
            for sequence in batch:
                self.assertEqual(counter, int(sequence.id))
                counter += 1

        self.assertEqual(307, counter)

        generator = sequence_generator.build_generator(110)

        counter = 0

        for batch in generator:
            for sequence in batch:
                self.assertEqual(counter, int(sequence.id))
                counter += 1

        self.assertEqual(307, counter)

        shutil.rmtree(path)
