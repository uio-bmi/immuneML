import os
import shutil
from unittest import TestCase

import numpy as np

from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestRepertoire(TestCase):

    def test_repertoire(self):

        path = EnvironmentSettings.tmp_test_path / "sequencerepertoire/"
        PathBuilder.build(path)

        sequences = [ReceptorSequence(amino_acid_sequence="AAA", identifier="1",
                                      metadata=SequenceMetadata(v_gene="V1", cell_id="1", chain=Chain.ALPHA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(amino_acid_sequence="CCC", identifier="2",
                                      metadata=SequenceMetadata(j_gene="J1", cell_id="1", chain=Chain.BETA,
                                                                custom_params={"cmv": "yes", "coeliac": True}))]

        obj = Repertoire.build_from_sequence_objects(sequences, path, {"cmv": "yes", 'subject_id': "1"})

        self.assertTrue(os.path.isfile(obj.data_filename))
        self.assertTrue(isinstance(obj, Repertoire))
        self.assertTrue(np.array_equal(np.array(["1", "2"]), obj.get_sequence_identifiers()))
        self.assertTrue(np.array_equal(np.array(["AAA", "CCC"]), obj.get_sequence_aas()))
        self.assertTrue(np.array_equal(np.array(["V1", None]), obj.get_v_genes()))
        self.assertTrue(np.array_equal(np.array([None, "J1"]), obj.get_j_genes()))
        self.assertTrue(np.array_equal(np.array(["no", "yes"]), obj.get_attribute("cmv")))
        self.assertTrue(np.array_equal(np.array([False, True]), obj.get_attribute("coeliac")))
        self.assertEqual("yes", obj.metadata["cmv"])
        self.assertEqual("1", obj.metadata["subject_id"])

        rebuilt_sequences = obj.sequences

        self.assertTrue(all(isinstance(seq, ReceptorSequence) for seq in rebuilt_sequences))
        self.assertEqual(2, len(rebuilt_sequences))
        self.assertEqual("1", rebuilt_sequences[0].identifier)
        self.assertEqual("2", rebuilt_sequences[1].identifier)
        self.assertEqual("AAA", rebuilt_sequences[0].amino_acid_sequence)
        self.assertEqual("yes", rebuilt_sequences[1].metadata.custom_params["cmv"])

        obj.free_memory()

        self.assertTrue(key in obj.data for key in Repertoire.FIELDS)
        self.assertTrue(obj.data[key] is None for key in Repertoire.FIELDS)

        shutil.rmtree(path)

    def test_receptor(self):
        path = EnvironmentSettings.tmp_test_path / "receptortestingpathrepertoire/"
        PathBuilder.build(path)

        sequences = [ReceptorSequence(amino_acid_sequence="AAA", identifier="1",
                                      metadata=SequenceMetadata(v_gene="V1", cell_id="1", chain=Chain.ALPHA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(amino_acid_sequence="CCC", identifier="2",
                                      metadata=SequenceMetadata(j_gene="J1", cell_id="1", chain=Chain.BETA,
                                                                custom_params={"cmv": "yes", "coeliac": True})),
                     ReceptorSequence(amino_acid_sequence="FFF", identifier="3",
                                      metadata=SequenceMetadata(v_gene="V1", cell_id="1", chain=Chain.ALPHA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(amino_acid_sequence="EEE", identifier="4",
                                      metadata=SequenceMetadata(j_gene="J1", cell_id="1", chain=Chain.BETA,
                                                                custom_params={"cmv": "yes", "coeliac": True})),
                     ReceptorSequence(amino_acid_sequence="FFF", identifier="5",
                                      metadata=SequenceMetadata(v_gene="V1", cell_id="2", chain=Chain.GAMMA,
                                                                custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(amino_acid_sequence="EEE", identifier="6",
                                      metadata=SequenceMetadata(j_gene="J1", cell_id="2", chain=Chain.DELTA,
                                                                custom_params={"cmv": "yes", "coeliac": True})),
                     ReceptorSequence(amino_acid_sequence="EEE", identifier="7",
                                      metadata=SequenceMetadata(j_gene="J2", cell_id="2", chain=Chain.DELTA,
                                                                custom_params={"cmv": "yes", "coeliac": True}))
                     ]

        obj = Repertoire.build_from_sequence_objects(sequences, path, {"cmv": "yes", 'subject_id': "1"})
        receptors = obj.receptors

        self.assertEqual(6, len(receptors))

        cells = obj.cells

        self.assertEqual(2, len(cells))

        shutil.rmtree(path)
