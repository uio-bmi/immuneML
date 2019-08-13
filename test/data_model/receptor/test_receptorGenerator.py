import pickle
import shutil
from unittest import TestCase

from source.data_model.receptor.BCReceptor import BCReceptor
from source.data_model.receptor.ReceptorGenerator import ReceptorGenerator
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestReceptorGenerator(TestCase):
    def test_build_generator(self):
        path = EnvironmentSettings.tmp_test_path + "receptor_generator/"
        PathBuilder.build(path)
        receptors = [BCReceptor(id=str(i)) for i in range(307)]
        file_list = ["{}batch{}.pkl".format(path, i) for i in range(4)]

        for i in range(4):
            with open(file_list[i], "wb") as file:
                pickle.dump(receptors[i * 100: (i+1) * 100], file)

        receptor_generator = ReceptorGenerator(file_list)
        generator = receptor_generator.build_generator(41)

        counter = 0

        for batch in generator:
            for receptor in batch:
                self.assertEqual(counter, int(receptor.id))
                self.assertTrue(isinstance(receptor, BCReceptor))
                counter += 1

        self.assertEqual(307, counter)

        generator = receptor_generator.build_generator(110)

        counter = 0

        for batch in generator:
            for receptor in batch:
                self.assertEqual(counter, int(receptor.id))
                self.assertTrue(isinstance(receptor, BCReceptor))
                counter += 1

        self.assertEqual(307, counter)

        shutil.rmtree(path)

