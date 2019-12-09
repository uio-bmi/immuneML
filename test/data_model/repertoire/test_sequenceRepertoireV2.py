import os
import pickle
import shutil
from time import time
from unittest import TestCase

import numpy as np

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.data_model.repertoire.SequenceRepertoireV2 import SequenceRepertoireV2
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceRepertoireV2(TestCase):

    def run_performance_test_for_v2(self, sequences, path, rep_id):
        obj = SequenceRepertoireV2.build_from_sequence_objects(sequences, path, rep_id, {"cmv": "yes"})

        load_start = time()

        obj._load()

        load_end = time()

        print("Loading from disk for repertoire with {} sequences took {}s.".format(len(sequences), load_end - load_start))

    def run_performance_test_for_old_version(self, sequences, path, rep_id):
        rep = SequenceRepertoire(sequences, {"cmv": "yes"}, rep_id)
        with open(path + "rep.pickle", "wb") as file:
            pickle.dump(rep, file)

        load_start = time()

        with open(path + "rep.pickle", "rb") as file:
            rep2 = pickle.load(file)

        load_end = time()

        print("Loading from disk for repertoire with {} (old version) sequences took {}s.".format(len(sequences), load_end - load_start))

    def create_sequences(self, count):
        return [ReceptorSequence(amino_acid_sequence="AAA", identifier=str(i),
                                 metadata=SequenceMetadata(v_gene="V1", j_gene="J2", count=231,
                                                           custom_params={"cmv": "no", "coeliac": False}))
                for i in range(count)]

    def test_performance(self):
        path = EnvironmentSettings.tmp_test_path + "sequencerepertoirev2performance/"
        PathBuilder.build(path)

        sequences = self.create_sequences(50000)

        self.run_performance_test_for_v2(sequences, path, "1")
        self.run_performance_test_for_old_version(sequences, path, "2")

        sequences = self.create_sequences(100000)

        self.run_performance_test_for_v2(sequences, path, "3")
        self.run_performance_test_for_old_version(sequences, path, "4")

        sequences = self.create_sequences(200000)

        self.run_performance_test_for_v2(sequences, path, "5")
        self.run_performance_test_for_old_version(sequences, path, "6")

        shutil.rmtree(path)

    def test_sequence_repertoire(self):

        path = EnvironmentSettings.tmp_test_path + "sequencerepertoirev2/"
        PathBuilder.build(path)

        sequences = [ReceptorSequence(amino_acid_sequence="AAA", identifier="1",
                                      metadata=SequenceMetadata(v_gene="V1", custom_params={"cmv": "no", "coeliac": False})),
                     ReceptorSequence(amino_acid_sequence="CCC", identifier="2",
                                      metadata=SequenceMetadata(j_gene="J1", custom_params={"cmv": "yes", "coeliac": True}))]

        obj = SequenceRepertoireV2.build_from_sequence_objects(sequences, path, "1", {"cmv": "yes"})

        self.assertTrue(os.path.isfile(obj._filename))
        self.assertTrue(isinstance(obj, SequenceRepertoireV2))
        self.assertTrue(np.array_equal(np.array(["1", "2"]), obj.get_sequence_identifiers()))
        self.assertTrue(np.array_equal(np.array(["AAA", "CCC"]), obj.get_sequence_aas()))
        self.assertTrue(np.array_equal(np.array(["V1", None]), obj.get_v_genes()))
        self.assertTrue(np.array_equal(np.array([None, "J1"]), obj.get_j_genes()))
        self.assertTrue(np.array_equal(np.array(["no", "yes"]), obj.get_custom_attribute("cmv")))
        self.assertTrue(np.array_equal(np.array([False, True]), obj.get_custom_attribute("coeliac")))
        self.assertEqual("yes", obj.metadata["cmv"])
        self.assertEqual("1", obj.identifier)

        rebuilt_sequences = obj.sequences

        self.assertTrue(all(isinstance(seq, ReceptorSequence) for seq in rebuilt_sequences))
        self.assertEqual(2, len(rebuilt_sequences))
        self.assertEqual("1", rebuilt_sequences[0].identifier)
        self.assertEqual("2", rebuilt_sequences[1].identifier)
        self.assertEqual("AAA", rebuilt_sequences[0].amino_acid_sequence)
        self.assertEqual("yes", rebuilt_sequences[1].metadata.custom_params["cmv"])

        obj.free_memory()

        self.assertTrue(key in obj.data for key in SequenceRepertoireV2.FIELDS)
        self.assertTrue(obj.data[key] is None for key in SequenceRepertoireV2.FIELDS)

        shutil.rmtree(path)
