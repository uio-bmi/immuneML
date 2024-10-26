import shutil
from unittest import TestCase

import pandas as pd

from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestRepertoireBuilder(TestCase):
    def test_build(self):
        path = EnvironmentSettings.tmp_test_path / "repbuilder/"
        repertoires, metadata = RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path / "rep1", {"default": [1, 2]})

        self.assertEqual(2, len(repertoires))
        self.assertEqual((2, 4), pd.read_csv(metadata).shape)

        self.assertEqual(2, len(repertoires[0].sequences()))
        self.assertTrue(all([isinstance(seq, ReceptorSequence) for seq in repertoires[0].sequences()]))
        self.assertEqual(1, repertoires[0].metadata["default"])

        self.assertEqual(1, len(repertoires[1].sequences()))
        self.assertTrue(all([isinstance(seq, ReceptorSequence) for seq in repertoires[1].sequences()]))
        self.assertEqual(2, repertoires[1].metadata["default"])
        self.assertEqual("rep_1", repertoires[1].metadata["subject_id"])

        # Testing with custom metadata
        repertoires, metadata = RepertoireBuilder.build([["AAA", "CCC"]], path / "rep2",
                                                        seq_metadata=[[{"v_call": "v5", "j_call": "j5"}, {"v_call": "v2", "j_call": "j2"}]])

        self.assertEqual(repertoires[0].sequences[0].metadata.v_call, "v5")
        self.assertEqual(repertoires[0].sequences[0].metadata.j_call, "j5")
        self.assertEqual(repertoires[0].sequences[1].metadata.v_call, "v2")
        self.assertEqual(repertoires[0].sequences[1].metadata.j_call, "j2")

        shutil.rmtree(path)

