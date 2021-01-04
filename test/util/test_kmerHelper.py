import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.KmerHelper import KmerHelper
from source.util.PathBuilder import PathBuilder


class TestKmerHelper(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_create_all_kmers(self):
        alphabet = list("ABCD")
        k = 2
        kmers = KmerHelper.create_all_kmers(k=k, alphabet=alphabet)
        self.assertEqual(len(kmers), 16)
        self.assertTrue("BD" in kmers)
        self.assertTrue("DA" in kmers)

    def test_create_kmers_from_sequence(self):
        kmers = KmerHelper.create_kmers_from_sequence(ReceptorSequence(amino_acid_sequence="ABCDEFG"), 3)
        self.assertTrue("ABC" in kmers and "BCD" in kmers and "CDE" in kmers and "DEF" in kmers and "EFG" in kmers)
        self.assertEqual(5, len(kmers))

        kmers = KmerHelper.create_kmers_from_sequence(ReceptorSequence(amino_acid_sequence="AB"), 3)
        self.assertTrue(len(kmers) == 0)

    def test_create_sentences_from_repertoire(self):

        path = EnvironmentSettings.tmp_test_path / "kmer/"
        PathBuilder.build(path)

        rep = Repertoire.build_from_sequence_objects([ReceptorSequence(amino_acid_sequence="AACT"),
                                                      ReceptorSequence(amino_acid_sequence="ACCT"),
                                                      ReceptorSequence(amino_acid_sequence="AACT")], path, {})

        sentences = KmerHelper.create_sentences_from_repertoire(rep, 3)

        self.assertEqual(3, len(sentences))
        self.assertTrue(len(sentences[0]) == 2 and "AAC" in sentences[0] and "ACT" in sentences[0])

        shutil.rmtree(path)

    def test_create_kmers_within_HD(self):

        kmers = KmerHelper.create_kmers_within_HD("ACT", list("ACTEF"), 1)

        self.assertEqual(15, len(kmers))
        for i in range(15):
            self.assertTrue(set("ACT").intersection(set(kmers[i][1])))

    def test_create_kmers_from_string(self):
        kmers = KmerHelper.create_kmers_from_string("ABCDEFG", 3)
        self.assertTrue("ABC" in kmers and "BCD" in kmers and "CDE" in kmers and "DEF" in kmers and "EFG" in kmers)
        self.assertEqual(5, len(kmers))

        kmers = KmerHelper.create_kmers_from_string("AB", 3)
        self.assertTrue(len(kmers) == 0)

    def test_create_IMGT_kmers_from_sequence(self):
        kmers = KmerHelper.create_IMGT_kmers_from_sequence(ReceptorSequence("CASSRYUF"), 3)
        self.assertTrue(("CAS", 105) in kmers)
        self.assertTrue(("ASS", 106) in kmers)
        self.assertTrue(("SSR", 107) in kmers)
        self.assertTrue(("SRY", 108) in kmers)
        self.assertTrue(("RYU", 114) in kmers)
        self.assertTrue(("YUF", 115) in kmers)

    def test_create_IMGT_gapped_kmers_from_sequence(self):
        kmers = KmerHelper.create_IMGT_gapped_kmers_from_sequence(ReceptorSequence("CASSRYUF"), 2, 1, 1, 1)
        self.assertTrue(all([k in kmers for k in [('CA.S', 105), ('AS.R', 106), ('SS.Y', 107), ('SR.U', 108), ('RY.F', 114)]]))

    def test_create_gapped_kmers_from_string(self):
        kmers = KmerHelper.create_gapped_kmers_from_string("CASSRYUF", 2, 1, 1, 1)
        self.assertTrue(all([k in kmers for k in ['CA.S', 'AS.R', 'SS.Y', 'SR.U', 'RY.F']]))
