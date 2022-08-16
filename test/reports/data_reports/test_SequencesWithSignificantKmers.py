import os
import shutil
from pathlib import Path
from unittest import TestCase
import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.SequencesWithSignificantKmers import SequencesWithSignificantKmers
from immuneML.reports.data_reports.SignificantKmerPositions import SignificantKmerPositions
from immuneML.util.PathBuilder import PathBuilder


class TestSequencesWithSignificantKmers(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _get_example_dataset(self, path):
        rep1 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(amino_acid_sequence="AAA", identifier="1"),
                              ReceptorSequence(amino_acid_sequence="III", identifier="2"),
                              ReceptorSequence(amino_acid_sequence="GGGG", identifier="3"),
                              ReceptorSequence(amino_acid_sequence="MMM", identifier="4")],
            path=path, metadata={"mylabel": "+"})
        rep2 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(amino_acid_sequence="IAIAA", identifier="1"),
                              ReceptorSequence(amino_acid_sequence="GGGG", identifier="3"),
                              ReceptorSequence(amino_acid_sequence="MMM", identifier="4")],
            path=path, metadata={"mylabel": "+"})
        rep21 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(amino_acid_sequence="IAIAA", identifier="1"),
                              ReceptorSequence(amino_acid_sequence="GGGG", identifier="3"),
                              ReceptorSequence(amino_acid_sequence="MMM", identifier="4")],
            path=path, metadata={"mylabel": "+"})
        rep22 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(amino_acid_sequence="IAIAA", identifier="1"),
                              ReceptorSequence(amino_acid_sequence="IIII", identifier="3"),
                              ReceptorSequence(amino_acid_sequence="IIII", identifier="4")],
            path=path, metadata={"mylabel": "-"})
        rep23 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(amino_acid_sequence="AAAAA", identifier="1"),
                              ReceptorSequence(amino_acid_sequence="IIII", identifier="3"),
                              ReceptorSequence(amino_acid_sequence="IIII", identifier="4")],
            path=path, metadata={"mylabel": "-"})
        rep3 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(amino_acid_sequence="KKKK", identifier="5"),
                              ReceptorSequence(amino_acid_sequence="HHH", identifier="6"),
                              ReceptorSequence(amino_acid_sequence="AAAA", identifier="7"),
                              ReceptorSequence(amino_acid_sequence="IIII", identifier="8")],
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
        path = EnvironmentSettings.root_path / f"test/tmp/significant_kmer_positions/"

        PathBuilder.build(path)

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

        self.assertEqual(result.output_tables[0].path, Path(path / "sequences_with_significant_2-mers_at_p=1.0.txt"))
        self.assertEqual(result.output_tables[1].path, Path(path / "sequences_with_significant_2-mers_at_p=0.1.txt"))
        self.assertEqual(result.output_tables[2].path, Path(path / "sequences_with_significant_3-mers_at_p=1.0.txt"))
        self.assertEqual(result.output_tables[3].path, Path(path / "sequences_with_significant_3-mers_at_p=0.1.txt"))


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


