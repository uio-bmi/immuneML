import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.data_model.SequenceParams import Chain, RegionType
from immuneML.data_model.SequenceSet import ReceptorSequence, Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestRepertoire(TestCase):

    def test_repertoire(self):
        path = EnvironmentSettings.tmp_test_path / "sequencerepertoire/"
        PathBuilder.remove_old_and_build(path)

        sequences = [ReceptorSequence(sequence_aa="AAA", sequence_id="1",
                                      v_call="V1*01", cell_id="1", locus=Chain.ALPHA.name,
                                      metadata={"cmv": "no", "coeliac": False}),
                     ReceptorSequence(sequence_aa="CCC", sequence_id="2",
                                      j_call="J1*01", cell_id="1", locus=Chain.BETA.name,
                                      metadata={"cmv": "yes", "coeliac": True})]

        obj = Repertoire.build_from_sequences(sequences, path, "rep2", {"cmv": "yes", 'subject_id': "1"})

        data = obj.data

        self.assertTrue(os.path.isfile(obj.data_filename))
        self.assertTrue(isinstance(obj, Repertoire))
        self.assertTrue(np.array_equal(np.array(["1", "2"]), data.sequence_id.tolist()))
        self.assertTrue(np.array_equal(np.array(["AAA", "CCC"]), data.cdr3_aa.tolist()))
        self.assertTrue(np.array_equal(np.array(["V1*01", '']), data.v_call.tolist()))
        self.assertTrue(np.array_equal(np.array(['', "J1*01"]), data.j_call.tolist()))
        self.assertTrue(np.array_equal(np.array(["no", "yes"]), data.cmv.tolist()))
        self.assertTrue(np.array_equal(np.array([False, True]), data.coeliac.tolist()))
        self.assertEqual("yes", obj.metadata["cmv"])
        self.assertEqual("1", obj.metadata["subject_id"])

        rebuilt_sequences = obj.sequences(RegionType.IMGT_CDR3)

        self.assertTrue(all(isinstance(seq, ReceptorSequence) for seq in rebuilt_sequences))
        self.assertEqual(2, len(rebuilt_sequences))
        self.assertEqual("1", rebuilt_sequences[0].sequence_id)
        self.assertEqual("2", rebuilt_sequences[1].sequence_id)
        self.assertEqual("AAA", rebuilt_sequences[0].sequence_aa)
        self.assertEqual("yes", rebuilt_sequences[1].metadata["cmv"])

        shutil.rmtree(path)

    def test_receptor(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "build_rep_from_receptors/")

        sequences = [
            ReceptorSequence(sequence_aa="AAA", sequence_id="1", v_call="V1", cell_id="1", locus=Chain.ALPHA.name,
                             metadata={"cmv": "no", "coeliac": False}),
            ReceptorSequence(sequence_aa="CCC", sequence_id="2",
                             j_call="J1", cell_id="1", locus=Chain.BETA.name,
                             metadata={"cmv": "yes", "coeliac": True}),
            ReceptorSequence(sequence_aa="FFF", sequence_id="3",
                             v_call="V1", cell_id="2", locus=Chain.ALPHA.name,
                             metadata={"cmv": "no", "coeliac": False}),
            ReceptorSequence(sequence_aa="EEE", sequence_id="4",
                             j_call="J1", cell_id="2", locus=Chain.BETA.name,
                             metadata={"cmv": "yes", "coeliac": True}),
            ReceptorSequence(sequence_aa="FFF", sequence_id="5",
                             v_call="V1", cell_id="3", locus=Chain.GAMMA.name,
                             metadata={"cmv": "no", "coeliac": False}),
            ReceptorSequence(sequence_aa="EEE", sequence_id="6",
                             j_call="J1", cell_id="3", locus=Chain.DELTA.name,
                             metadata={"cmv": "yes", "coeliac": True})
            ]

        obj = Repertoire.build_from_sequences(sequences, path, "rep1", {"cmv": "yes", 'subject_id': "1"},
                                              RegionType.IMGT_CDR3)
        receptors = obj.receptors(RegionType.IMGT_CDR3)

        self.assertEqual(3, len(receptors))

        shutil.rmtree(path)
