import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.RecoveredSignificantFeatures import RecoveredSignificantFeatures
from immuneML.util.PathBuilder import PathBuilder


class TestSignificantFeatures(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _get_example_dataset(self, path):
        rep1 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="AAA", sequence_id="1", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="III", sequence_id="2", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="GGGG", sequence_id="3", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="MMM", sequence_id="4", metadata=SequenceMetadata(region_type="IMGT_CDR3"))],
            path=path, metadata={"mylabel": "+"})
        rep2 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="GGGG", sequence_id="3", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="MMM", sequence_id="4", metadata=SequenceMetadata(region_type="IMGT_CDR3"))],
            path=path, metadata={"mylabel": "+"})
        rep21 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="GGGG", sequence_id="3", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="MMM", sequence_id="4", metadata=SequenceMetadata(region_type="IMGT_CDR3"))],
            path=path, metadata={"mylabel": "+"})
        rep22 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="3", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="4", metadata=SequenceMetadata(region_type="IMGT_CDR3"))],
            path=path, metadata={"mylabel": "-"})
        rep23 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="IAIAA", sequence_id="1", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="3", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="4", metadata=SequenceMetadata(region_type="IMGT_CDR3"))],
            path=path, metadata={"mylabel": "-"})
        rep3 = Repertoire.build_from_sequence_objects(
            sequence_objects=[ReceptorSequence(sequence_aa="KKKK", sequence_id="5", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="HHH", sequence_id="6", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="AAAA", sequence_id="7", metadata=SequenceMetadata(region_type="IMGT_CDR3")),
                              ReceptorSequence(sequence_aa="IIII", sequence_id="8", metadata=SequenceMetadata(region_type="IMGT_CDR3"))],
            path=path, metadata={"mylabel": "-"})

        dataset = RepertoireDataset(repertoires=[rep1, rep2, rep21, rep22, rep23, rep3],
                                    labels={"mylabel": ["+", "-"]})

        return dataset

    def _get_implanted_sequences(self, path):
        file_path = path / "sequences.txt"

        with open(file_path, "w") as f:
            f.writelines(["MMM\nGGG"])

        return str(file_path)

    def test_generate_with_compairr(self):
        working = 0
        for compairr_path in EnvironmentSettings.compairr_paths:
            if compairr_path.exists():
                working += 1
                self.test_generate(str(compairr_path))
                break

        assert working > 0

    def test_generate(self, compairr_path=None):
        path_suffix = "compairr" if compairr_path else "no_compairr"
        base_path = EnvironmentSettings.tmp_test_path / f"recovered_significant_features/"
        path = base_path / path_suffix

        PathBuilder.remove_old_and_build(path)

        dataset = self._get_example_dataset(path)
        implanted_sequences_path = self._get_implanted_sequences(path)

        report = RecoveredSignificantFeatures.build_object(**{"dataset": dataset,
                                                              "p_values": [0.5, 0.0],
                                                              "k_values": ["full_sequence", 3],
                                                              "compairr_path": compairr_path,
                                                              "groundtruth_sequences_path": implanted_sequences_path,
                                                              "label": {"mylabel": {"positive_class": "+"}},
                                                              "result_path": path,
                                                              "trim_leading_trailing": False})

        self.assertListEqual(report.groundtruth_sequences, ["MMM", "GGG"])

        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(len(result.output_figures), 2)
        self.assertEqual(len(result.output_tables), 1)

        self.assertEqual(result.output_figures[0].path, path / "n_significant_features_figure.html")
        self.assertEqual(result.output_figures[1].path, path / "n_true_features_figure.html")
        self.assertEqual(result.output_tables[0].path, path / "recovered_significant_features_report.csv")

        self.assertTrue(os.path.isfile(result.output_figures[0].path))
        self.assertTrue(os.path.isfile(result.output_figures[1].path))
        self.assertTrue(os.path.isfile(result.output_tables[0].path))

        result_output = pd.read_csv(path / "recovered_significant_features_report.csv", sep=",")

        self.assertListEqual(list(result_output.columns), ["encoding", "p-value", "n_significant", "n_true", "n_intersect"])
        self.assertListEqual(list(result_output["encoding"]), ["full_sequence"] * 2 + ["3-mer"] * 2)
        self.assertListEqual(list(result_output["p-value"]), [0.5, 0., 0.5, 0.])
        self.assertListEqual(list(result_output["n_significant"]), [2, 0] * 2)
        self.assertListEqual(list(result_output["n_true"]), [2, 2, 2, 2])
        self.assertListEqual(list(result_output["n_intersect"]), [1, 0, 2, 0])

        shutil.rmtree(base_path)
