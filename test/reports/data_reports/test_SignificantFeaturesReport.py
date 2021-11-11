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
from immuneML.reports.data_reports.SignificantFeaturesReport import SignificantFeaturesReport
from immuneML.util.PathBuilder import PathBuilder


class TestSignificantFeaturesReport(TestCase):

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
            sequence_objects=[ReceptorSequence(amino_acid_sequence="IAIAA", identifier="1"),
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

    def test_generate_with_compairr(self):
        compairr_paths = [Path("/usr/local/bin/compairr"), Path("./compairr/src/compairr")]

        for compairr_path in compairr_paths:
            if compairr_path.exists():
                self.test_generate(str(compairr_path))
                break

    def test_generate(self, compairr_path=None):
        path_suffix = "compairr" if compairr_path else "no_compairr"
        base_path = EnvironmentSettings.root_path / f"test/tmp/significant_features/"
        path = base_path / path_suffix

        PathBuilder.build(path)

        dataset = self._get_example_dataset(path)

        report = SignificantFeaturesReport.build_object(**{"dataset": dataset,
                                                           "p_values": [0.5, 0.0],
                                                           "k_values": ["full_sequence", 3],
                                                           "compairr_path": compairr_path,
                                                           "label": {"mylabel": {"positive_class": "+"}},
                                                           "result_path": path})

        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(len(result.output_figures), 1)
        self.assertEqual(len(result.output_tables), 1)

        self.assertEqual(result.output_figures[0].path, Path(path / "significant_features_figure.html"))
        self.assertEqual(result.output_tables[0].path, Path(path / "significant_features_report.csv"))

        self.assertTrue(os.path.isfile(result.output_figures[0].path))
        self.assertTrue(os.path.isfile(result.output_tables[0].path))

        result_output = pd.read_csv(path / "significant_features_report.csv", sep=",")

        self.assertListEqual(list(result_output.columns), ["encoding","p-value","class","significant_features"])
        self.assertListEqual(list(result_output["encoding"]), ["full_sequence"] * 12 + ["3-mer"] * 12)
        self.assertListEqual(list(result_output["p-value"]), [0.5] * 6 + [0.] * 6 + [0.5] * 6 + [0.] * 6)
        self.assertListEqual(list(result_output["class"]), ["+", "+", "+", "-", "-", "-"] * 4)
        self.assertListEqual(list(result_output["significant_features"]), [2.] * 3 + [0.] * 9 + [2.] * 3 + [0.] * 9)

        shutil.rmtree(base_path)


