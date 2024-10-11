import os
import shutil
from unittest import TestCase

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import Repertoire, ReceptorSequence
from immuneML.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceMatcher(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_match(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seqmatch/")

        repertoire = Repertoire.build_from_sequences(sequences=[
            ReceptorSequence(sequence_aa="AAAAAA", locus="A", v_call="V1", j_call="J2",
                             sequence_id="3"),
            ReceptorSequence(sequence_aa="CCCCCC", locus="A", v_call="V1", j_call="J2",
                             sequence_id="4"),
            ReceptorSequence(sequence_aa="AAAACC", locus="A", v_call="V1", j_call="J2",
                             sequence_id="5"),
            ReceptorSequence(sequence_aa="TADQVF", locus="A", v_call="V1", j_call="J3",
                             sequence_id="6")],
            metadata={"CD": True}, result_path=path)

        dataset = RepertoireDataset(repertoires=[repertoire])
        sequences = [ReceptorSequence(sequence_aa="AAAACA", locus="A", v_call="V1", j_call="J2",
                                      sequence_id="1"),
                     ReceptorSequence(sequence_aa="TADQV", locus="A", v_call="V1", j_call="J3", sequence_id="2")]

        matcher = SequenceMatcher()
        result = matcher.match(dataset, sequences, 2, SequenceMatchingSummaryType.PERCENTAGE)

        self.assertTrue("repertoires" in result)
        self.assertEqual(1, len(result["repertoires"][0]["sequences"][3]["matching_sequences"]))
        self.assertTrue(result["repertoires"][0]["metadata"]["CD"])
        self.assertEqual(1, len(result["repertoires"]))

        shutil.rmtree(path)

    def test_match_repertoire(self):
        path = EnvironmentSettings.tmp_test_path / "seqmatchrep/"
        PathBuilder.remove_old_and_build(path)

        seq_objs = [ReceptorSequence(sequence_aa="AAAAAA", sequence_id="1", locus="A", duplicate_count=3),
                    ReceptorSequence(sequence_aa="CCCCCC", sequence_id="2", locus="A", duplicate_count=2),
                    ReceptorSequence(sequence_aa="AAAACC", sequence_id="3", locus="A", duplicate_count=1),
                    ReceptorSequence(sequence_aa="TADQVF", sequence_id="4", locus="A", duplicate_count=4)]

        repertoire = Repertoire.build_from_sequences(sequences=seq_objs, metadata={"CD": True}, result_path=path)

        sequences = [ReceptorSequence(sequence_aa="AAAACA", locus="A"),
                     ReceptorSequence(sequence_aa="TADQV", locus="A")]

        matcher = SequenceMatcher()
        result = matcher.match_repertoire(repertoire, 0, sequences, 2, SequenceMatchingSummaryType.COUNT)

        self.assertTrue("sequences" in result)
        self.assertTrue("repertoire" in result)
        self.assertTrue("repertoire_index" in result)

        self.assertEqual(4, len(result["sequences"]))
        self.assertEqual(1, len(result["sequences"][0]["matching_sequences"]))
        self.assertEqual(0, len(result["sequences"][1]["matching_sequences"]))
        self.assertEqual(1, len(result["sequences"][2]["matching_sequences"]))
        self.assertEqual(1, len(result["sequences"][3]["matching_sequences"]))

        self.assertEqual(3, len([r for r in result["sequences"] if len(r["matching_sequences"]) > 0]))
        self.assertTrue(result["metadata"]["CD"])

        result = matcher.match_repertoire(repertoire, 0, sequences, 2, SequenceMatchingSummaryType.CLONAL_PERCENTAGE)
        self.assertEqual(0.8, result["clonal_percentage"])

        shutil.rmtree(path)
