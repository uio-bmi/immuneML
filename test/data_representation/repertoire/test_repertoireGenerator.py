import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireGenerator import RepertoireGenerator
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.util.PathBuilder import PathBuilder


class TestRepertoireGenerator(TestCase):
    def test_build_generator(self):

        # prepare data
        sequence1 = ReceptorSequence(amino_acid_sequence="CAS")
        sequence2 = ReceptorSequence(amino_acid_sequence="VFA")

        repertoire1 = Repertoire([sequence2, sequence1], RepertoireMetadata())
        repertoire2 = Repertoire([sequence1, sequence2], RepertoireMetadata())

        path = "./repertoire_generator/"
        file1 = path + "rep1"
        file2 = path + "rep2"
        file3 = path + "rep3"

        PathBuilder.build(path)

        with open(file1, "wb") as file:
            pickle.dump(repertoire1, file)
        with open(file2, "wb") as file:
            pickle.dump(repertoire2, file)
        with open(file3, "wb") as file:
            pickle.dump(repertoire1, file)

        # test generator behavior
        rep_gen1 = RepertoireGenerator.build_generator([file1, file2], 1)
        rep1 = next(rep_gen1)
        self.assertTrue(isinstance(rep1, Repertoire))
        self.assertEqual(len(rep1.sequences), 2)
        self.assertEqual(rep1.sequences[1].amino_acid_sequence, "CAS")
        self.assertEqual(rep1.sequences[0].amino_acid_sequence, "VFA")

        rep2 = next(rep_gen1)
        self.assertTrue(isinstance(rep2, Repertoire))
        self.assertEqual(len(rep2.sequences), 2)
        self.assertEqual(rep2.sequences[0].amino_acid_sequence, "CAS")
        self.assertEqual(rep2.sequences[1].amino_acid_sequence, "VFA")

        rep_gen2 = RepertoireGenerator.build_generator([file1, file2, file3], 2)

        i = 0

        for repertoire in rep_gen2:
            self.assertEqual(len(repertoire.sequences), 2)
            i = i + 1

        self.assertEqual(i, 3)

        # clean up
        shutil.rmtree(path)

