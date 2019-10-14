import pickle
import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.RepertoireBuilder import RepertoireBuilder


class TestRepertoireBuilder(TestCase):
    def test_build(self):
        path = EnvironmentSettings.root_path + "test/tmp/repbuilder/"
        filenames, metadata = RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path, {"default": [1, 2]})

        self.assertEqual(2, len(filenames))
        self.assertEqual((2, 3), pd.read_csv(metadata).shape)

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
        self.assertEqual("rep_1", rep2.identifier)


        # Testing with custom metadata
        filenames, metadata = RepertoireBuilder.build([["AAA", "CCC"]], path, seq_metadata=[[{"v_gene": "v5", "j_gene": "j5"}, {"v_gene": "v2", "j_gene": "j2"}]])

        with open(filenames[0], "rb") as file:
            rep3 = pickle.load(file)

        self.assertEqual(rep3.sequences[0].metadata.v_gene, "v5")
        self.assertEqual(rep3.sequences[0].metadata.j_gene, "j5")
        self.assertEqual(rep3.sequences[1].metadata.v_gene, "v2")
        self.assertEqual(rep3.sequences[1].metadata.j_gene, "j2")

        shutil.rmtree(path)

