import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestRepertoire(TestCase):

    def test_repertoire(self):

        path = EnvironmentSettings.tmp_test_path / "sequencerepertoire/"
        PathBuilder.build(path)

        sequences = [ReceptorSequence(sequence_aa="AAA", sequence_id="1",
                                      metadata=SequenceMetadata(v_call="V1*01", cell_id="1", chain=Chain.ALPHA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(sequence_aa="CCC", sequence_id="2",
                                      metadata=SequenceMetadata(j_call="J1*01", cell_id="1", chain=Chain.BETA,
                                                                custom_params={"cmv": "yes", "coeliac": True}))]

        obj = Repertoire.build_from_sequence_objects(sequences, path, {"cmv": "yes", 'subject_id': "1"})

        self.assertTrue(os.path.isfile(obj.data_filename))
        self.assertTrue(isinstance(obj, Repertoire))
        self.assertTrue(np.array_equal(np.array(["1", "2"]), obj.get_sequence_identifiers().tolist()))
        self.assertTrue(np.array_equal(np.array(["AAA", "CCC"]), obj.get_sequence_aas().tolist()))
        self.assertTrue(np.array_equal(np.array(["V1", '']), obj.get_v_genes()))
        self.assertTrue(np.array_equal(np.array(['', "J1"]), obj.get_j_genes()))
        self.assertTrue(np.array_equal(np.array(["no", "yes"]), obj.get_attribute("cmv").tolist()))
        self.assertTrue(np.array_equal(np.array([False, True]), obj.get_attribute("coeliac").tolist()))
        self.assertEqual("yes", obj.metadata["cmv"])
        self.assertEqual("1", obj.metadata["subject_id"])

        rebuilt_sequences = obj.sequences

        self.assertTrue(all(isinstance(seq, ReceptorSequence) for seq in rebuilt_sequences))
        self.assertEqual(2, len(rebuilt_sequences))
        self.assertEqual("1", rebuilt_sequences[0].sequence_id)
        self.assertEqual("2", rebuilt_sequences[1].sequence_id)
        self.assertEqual("AAA", rebuilt_sequences[0].sequence_aa)
        self.assertEqual("yes", rebuilt_sequences[1].metadata.custom_params["cmv"])

        self.assertTrue(key in obj.data for key in Repertoire.FIELDS)
        self.assertTrue(obj.data[key] is None for key in Repertoire.FIELDS)

        shutil.rmtree(path)

    def test_receptor(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "receptortestingpathrepertoire/")

        sequences = [ReceptorSequence(sequence_aa="AAA", sequence_id="1",
                                      metadata=SequenceMetadata(v_call="V1", cell_id="1", chain=Chain.ALPHA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(sequence_aa="CCC", sequence_id="2",
                                      metadata=SequenceMetadata(j_call="J1", cell_id="1", chain=Chain.BETA,
                                                                custom_params={"cmv": "yes", "coeliac": True})),
                     ReceptorSequence(sequence_aa="FFF", sequence_id="3",
                                      metadata=SequenceMetadata(v_call="V1", cell_id="2", chain=Chain.ALPHA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(sequence_aa="EEE", sequence_id="4",
                                      metadata=SequenceMetadata(j_call="J1", cell_id="2", chain=Chain.BETA,
                                                                custom_params={"cmv": "yes", "coeliac": True})),
                     ReceptorSequence(sequence_aa="FFF", sequence_id="5",
                                      metadata=SequenceMetadata(v_call="V1", cell_id="3", chain=Chain.GAMMA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(sequence_aa="EEE", sequence_id="6",
                                      metadata=SequenceMetadata(j_call="J1", cell_id="3", chain=Chain.DELTA,
                                                                custom_params={"cmv": "yes", "coeliac": True}))
                     ]

        obj = Repertoire.build_from_sequence_objects(sequences, path, {"cmv": "yes", 'subject_id': "1"})
        receptors = obj.receptors

        self.assertEqual(3, len(receptors))

        shutil.rmtree(path)
