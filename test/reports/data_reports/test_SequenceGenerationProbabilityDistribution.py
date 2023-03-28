import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.SequenceGenerationProbabilityDistribution import \
    SequenceGenerationProbabilityDistribution
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceGenerationProbabilityDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

        self.path = EnvironmentSettings.root_path / "test/tmp/datareports/pgen/"
        PathBuilder.build(self.path)

        self.label = "TESTDATA"
        self.sequences = ["CSAIGQGKGAFYGYTF", "CASSLDRVSASGANVLTF", "CASSVQPRSEVPNTGELFF"]
        self.v_genes = ["TRBV20-1", "TRBV4-1", "TRBV11-3"]
        self.j_genes = ["TRBJ1-2", "TRBJ2-6", "TRBJ2-2"]
        self.counts = [1, 2, 1]

        repertoire = Repertoire.build(self.sequences,
                                      v_genes=self.v_genes,
                                      j_genes=self.j_genes,
                                      region_types=[RegionType.IMGT_JUNCTION for _ in range(3)],
                                      counts=self.counts,
                                      path=self.path)

        self.repertoire_dataset = RepertoireDataset(repertoires=[repertoire])

        self.sgpd = SequenceGenerationProbabilityDistribution(self.repertoire_dataset, self.path,
                                                              default_sequence_label=self.label,
                                                              mark_implanted_labels=False)

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_get_pgen_distribution(self):
        result = self.sgpd.generate_report()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

    def test_load_dataset_dataframe(self):
        df = self.sgpd._load_dataset_dataframe()

        for column in ["sequence_aas", "v_genes", "j_genes", "repertoire", "label"]:
            self.assertIn(column, list(df.columns))

        self.assertEqual(len(df), len(self.sequences))

        self.assertCountEqual(list(df["sequence_aas"]), self.sequences)
        self.assertCountEqual(list(df["v_genes"]), self.v_genes)
        self.assertCountEqual(list(df["j_genes"]), self.j_genes)
        self.assertListEqual(list(df["label"]), [self.label]*len(self.sequences))
        self.assertTrue(all(rep == df["repertoire"][0] for rep in df["repertoire"]))

    def test_sequential_pgen_computation(self):
        self.sgpd.number_of_processes = 1
        df = self.sgpd._load_dataset_dataframe()
        df = self.sgpd._get_sequence_pgen(df)

        self.assertIn("pgen", df.columns)

        for pgen in df["pgen"]:
            self.assertGreaterEqual(pgen, 0)
            self.assertLess(pgen, 1)

    def test_parallelized_pgen_computation(self):
        self.sgpd.number_of_processes = 2
        df = self.sgpd._load_dataset_dataframe()
        df = self.sgpd._get_sequence_pgen(df)

        self.assertIn("pgen", df.columns)

        for pgen in df["pgen"]:
            self.assertGreaterEqual(pgen, 0)
            self.assertLess(pgen, 1)

    def test_count_sequences(self):
        df = self.sgpd._load_dataset_dataframe()
        df = self.sgpd._get_sequence_count(df)

        self.assertCountEqual(df["count"], self.counts)
