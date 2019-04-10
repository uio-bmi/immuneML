import pickle
import shutil
from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.RepertoireBuilder import RepertoireBuilder


class TestRepertoireBuilder(TestCase):
    def test_build(self):
        path = EnvironmentSettings.root_path + "test/mp/repbuilder/"
        filenames = RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path, {"default": [1, 2]})

        self.assertEqual(2, len(filenames))

        with open(filenames[0], "rb") as file:
            rep1 = pickle.load(file)

        self.assertEqual(2, len(rep1.sequences))
        self.assertTrue(all([isinstance(seq, ReceptorSequence) for seq in rep1.sequences]))
        self.assertEqual(1, rep1.metadata.custom_params["default"])

        with open(filenames[1], "rb") as file:
            rep2 = pickle.load(file)

        self.assertEqual(1, len(rep2.sequences))
        self.assertTrue(all([isinstance(seq, ReceptorSequence) for seq in rep2.sequences]))
        self.assertEqual(2, rep2.metadata.custom_params["default"])

        shutil.rmtree(path)
