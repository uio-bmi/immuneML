import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.SequencesWithSignificantKmers import SequencesWithSignificantKmers
from immuneML.util.PathBuilder import PathBuilder


class TestSequencesWithSignificantKmers(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _get_example_dataset(self, path):
        rep1 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="AAA", sequence_id="1"),
                              ReceptorSequence(sequence_aa="III", sequence_id="2"),
                              ReceptorSequence(sequence_aa="GGGG", sequence_id="3"),
                              ReceptorSequence(sequence_aa="MMM", sequence_id="4")],
            path=path, metadata={"mylabel": "+"})
        rep2 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1"),
                              ReceptorSequence(sequence_aa="GGGG", sequence_id="3"),
                              ReceptorSequence(sequence_aa="MMM", sequence_id="4")],
            path=path, metadata={"mylabel": "+"})
        rep21 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1"),
                              ReceptorSequence(sequence_aa="GGGG", sequence_id="3"),
                              ReceptorSequence(sequence_aa="MMM", sequence_id="4")],
            path=path, metadata={"mylabel": "+"})
        rep22 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1"),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="3"),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="4")],
            path=path, metadata={"mylabel": "-"})
        rep23 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="AAAAA", sequence_id="1"),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="3"),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="4")],
            path=path, metadata={"mylabel": "-"})
        rep3 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="KKKK", sequence_id="5"),
                              ReceptorSequence(sequence_aa="HHH", sequence_id="6"),
                              ReceptorSequence(sequence_aa="AAAA", sequence_id="7"),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="8")],
            path=path, metadata={"mylabel": "-"})

        dataset = RepertoireDataset(repertoires=[rep1, rep2, rep21, rep22, rep23, rep3],
                                    labels={"mylabel": ["+", "-"]})

        return dataset

    def _get_implanted_sequences(self, path):
        file_path = path / "sequences.txt"

        with open(file_path, "w") as f:
            f.writelines(["IAIAA\nGGGG"])

        return str(file_path)

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / f"seqs_with_significant_kmer_positions/"

        PathBuilder.remove_old_and_build(path)

        dataset = self._get_example_dataset(path)
        implanted_sequences_path = self._get_implanted_sequences(path)

        report = SequencesWithSignificantKmers.build_object(**{"dataset": dataset,
                                                               "p_values": [1.0, 0.1],
                                                               "k_values": [2, 3],
                                                               "reference_sequences_path": implanted_sequences_path,
                                                               "label": {"mylabel": {"positive_class": "+"}},
                                                               "result_path": path})

        self.assertListEqual(report.reference_sequences, ["IAIAA", "GGGG"])

        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(len(result.output_tables), 4)

        self.assertEqual(result.output_tables[0].path, path / "sequences_with_significant_2-mers_at_p=1.0.txt")
        self.assertEqual(result.output_tables[1].path, path / "sequences_with_significant_2-mers_at_p=0.1.txt")
        self.assertEqual(result.output_tables[2].path, path / "sequences_with_significant_3-mers_at_p=1.0.txt")
        self.assertEqual(result.output_tables[3].path, path / "sequences_with_significant_3-mers_at_p=0.1.txt")

        self.assertTrue(os.path.isfile(result.output_tables[0].path))
        self.assertTrue(os.path.isfile(result.output_tables[1].path))
        self.assertTrue(os.path.isfile(result.output_tables[2].path))
        self.assertTrue(os.path.isfile(result.output_tables[3].path))

        result_output = pd.read_csv(path / "sequences_with_significant_2-mers_at_p=1.0.txt", sep=",", header=None)
        self.assertListEqual(list(result_output[0]), ['IAIAA', 'GGGG'])

        result_output = pd.read_csv(path / "sequences_with_significant_2-mers_at_p=0.1.txt", sep=",", header=None)
        self.assertListEqual(list(result_output[0]), ['GGGG'])

        result_output = pd.read_csv(path / "sequences_with_significant_3-mers_at_p=1.0.txt", sep=",", header=None)
        self.assertListEqual(list(result_output[0]), ['IAIAA', 'GGGG'])

        result_output = pd.read_csv(path / "sequences_with_significant_3-mers_at_p=0.1.txt", sep=",", header=None)
        self.assertListEqual(list(result_output[0]), ['GGGG'])

        shutil.rmtree(path)
