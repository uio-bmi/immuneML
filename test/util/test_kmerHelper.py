import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.KmerHelper import KmerHelper
from immuneML.util.PathBuilder import PathBuilder


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
        kmers = KmerHelper.create_kmers_from_sequence(ReceptorSequence(sequence_aa="ABCDEFG"), 3, sequence_type=SequenceType.AMINO_ACID)
        self.assertTrue("ABC" in kmers and "BCD" in kmers and "CDE" in kmers and "DEF" in kmers and "EFG" in kmers)
        self.assertEqual(5, len(kmers))

        kmers = KmerHelper.create_kmers_from_sequence(ReceptorSequence(sequence_aa="AB"), 3, sequence_type=SequenceType.AMINO_ACID)
        self.assertTrue(len(kmers) == 0)

    def test_create_sentences_from_repertoire(self):

        path = EnvironmentSettings.tmp_test_path / "kmer/"
        PathBuilder.remove_old_and_build(path)

        rep = Repertoire.build_from_sequence_objects([ReceptorSequence(sequence_aa="AACT"),
                                                      ReceptorSequence(sequence_aa="ACCT"),
                                                      ReceptorSequence(sequence_aa="AACT")], path, {})

        sentences = KmerHelper.create_sentences_from_repertoire(rep, 3, sequence_type=SequenceType.AMINO_ACID)

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
        kmers = KmerHelper.create_IMGT_kmers_from_sequence(ReceptorSequence("CASSRYUF", metadata=SequenceMetadata(region_type="IMGT_CDR3")), 3, sequence_type=SequenceType.AMINO_ACID)
        self.assertTrue(("CAS", '105') in kmers)
        self.assertTrue(("ASS", '106') in kmers)
        self.assertTrue(("SSR", '107') in kmers)
        self.assertTrue(("SRY", '108') in kmers)
        self.assertTrue(("RYU", '114') in kmers)
        self.assertTrue(("YUF", '115') in kmers)

    def test_create_IMGT_gapped_kmers_from_sequence(self):
        kmers = KmerHelper.create_IMGT_gapped_kmers_from_sequence(ReceptorSequence("CASSRYUF", metadata=SequenceMetadata(region_type="IMGT_CDR3")), SequenceType.AMINO_ACID, 2, 1, 1, 1)
        self.assertTrue(all([k in kmers for k in [('CA.S', '105'), ('AS.R', '106'), ('SS.Y', '107'), ('SR.U', '108'), ('RY.F', '114')]]))

    def test_create_gapped_kmers_from_string(self):
        kmers = KmerHelper.create_gapped_kmers_from_string("CASSRYUF", 2, 1, 1, 1)
        self.assertTrue(all([k in kmers for k in ['CA.S', 'AS.R', 'SS.Y', 'SR.U', 'RY.F']]))
