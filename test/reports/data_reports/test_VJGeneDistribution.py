import os
import shutil
from unittest import TestCase

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.VJGeneDistribution import VJGeneDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestVJGeneDistribution(TestCase):
    def test_generate_sequence_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "overview_sequence_dataset/")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {10: 0.5, 11:0.25, 20:0.25}, {"l1": {"a": 0.5, "b": 0.5}}, path / "dataset")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "VJGeneDistribution")
        params["dataset"] = dataset
        params["label"] = "l1"
        params["result_path"] = path / "result"

        report = VJGeneDistribution.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "result/J_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/V_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/VJ_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/TRB_J_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_V_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_VJ_gene_distribution_l1=a.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_VJ_gene_distribution_l1=b.html"))

        shutil.rmtree(path)

    def test_generate_receptor_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "overview_receptor_dataset/")


        dataset = RandomDatasetGenerator.generate_receptor_dataset(100, chain_1_length_probabilities={10: 0.5, 11:0.25, 20:0.25},
                                                                   chain_2_length_probabilities={10: 0.5, 11: 0.25, 15: 0.25},
                                                                   labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / "dataset")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "VJGeneDistribution")
        params["dataset"] = dataset
        params["result_path"] = path / "result"

        report = VJGeneDistribution.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "result/J_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/V_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/VJ_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/TRA_J_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRA_V_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRA_VJ_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_J_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_V_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_VJ_gene_distribution.html"))

        shutil.rmtree(path)

    def test_generate_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "overview_repertoire_dataset/")


        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=5, sequence_count_probabilities={20:0.25, 30:0.25, 40:0.25, 50:0.25},
                                                                     sequence_length_probabilities={10: 1},
                                                                     labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / "dataset")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "VJGeneDistribution")
        params["dataset"] = dataset
        params["label"] = "l1"
        params["result_path"] = path / "result"

        report = VJGeneDistribution.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)


        self.assertTrue(os.path.isfile(path / "result/J_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/V_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/VJ_gene_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/VJ_gene_distribution_averaged_across_repertoires.tsv"))
        self.assertTrue(os.path.isfile(path / "result/TRB_J_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_V_gene_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_VJ_gene_distribution_l1=a_averaged_across_repertoires.html"))
        self.assertTrue(os.path.isfile(path / "result/TRB_VJ_gene_distribution_l1=b_averaged_across_repertoires.html"))

        shutil.rmtree(path)