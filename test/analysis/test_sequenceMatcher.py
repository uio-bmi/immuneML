import os
import shutil
from unittest import TestCase

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceMatcher(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_match(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seqmatch/")

        repertoire = Repertoire.build_from_sequence_objects(sequence_objects=[
            ReceptorSequence(sequence_aa="AAAAAA", metadata=SequenceMetadata(locus="A", v_call="V1", j_call="J2",
                                                                             region_type=RegionType.IMGT_CDR3.name),
                             sequence_id="3"),
            ReceptorSequence(sequence_aa="CCCCCC", metadata=SequenceMetadata(locus="A", v_call="V1", j_call="J2",
                                                                             region_type=RegionType.IMGT_CDR3.name),
                             sequence_id="4"),
            ReceptorSequence(sequence_aa="AAAACC", metadata=SequenceMetadata(locus="A", v_call="V1", j_call="J2",
                                                                             region_type=RegionType.IMGT_CDR3.name),
                             sequence_id="5"),
            ReceptorSequence(sequence_aa="TADQVF", metadata=SequenceMetadata(locus="A", v_call="V1", j_call="J3",
                                                                             region_type=RegionType.IMGT_CDR3.name),
                             sequence_id="6")],
            metadata={"CD": True}, path=path)

        dataset = RepertoireDataset(repertoires=[repertoire])
        sequences = [ReceptorSequence("AAAACA", metadata=SequenceMetadata(locus="A", v_call="V1", j_call="J2",
                                                                          region_type=RegionType.IMGT_CDR3.name),
                                      sequence_id="1"),
                     ReceptorSequence("TADQV", metadata=SequenceMetadata(locus="A", v_call="V1", j_call="J3",
                                                                         region_type=RegionType.IMGT_CDR3.name),
                                      sequence_id="2")]

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

        seq_objs = [ReceptorSequence(sequence_aa="AAAAAA", sequence_id="1",
                                     metadata=SequenceMetadata(locus="A", duplicate_count=3)),
                    ReceptorSequence(sequence_aa="CCCCCC", sequence_id="2",
                                     metadata=SequenceMetadata(locus="A", duplicate_count=2)),
                    ReceptorSequence(sequence_aa="AAAACC", sequence_id="3",
                                     metadata=SequenceMetadata(locus="A", duplicate_count=1)),
                    ReceptorSequence(sequence_aa="TADQVF", sequence_id="4",
                                     metadata=SequenceMetadata(locus="A", duplicate_count=4))]

        repertoire = Repertoire.build_from_sequence_objects(sequence_objects=seq_objs,
                                                            metadata={"CD": True}, path=path)

        sequences = [ReceptorSequence("AAAACA", metadata=SequenceMetadata(locus="A")),
                     ReceptorSequence("TADQV", metadata=SequenceMetadata(locus="A"))]

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
