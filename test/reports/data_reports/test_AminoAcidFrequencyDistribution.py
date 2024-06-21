import os
import shutil
import pandas as pd
from unittest import TestCase

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.AminoAcidFrequencyDistribution import AminoAcidFrequencyDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestAminoAcidFrequencyDistribution(TestCase):
    def test_generate_sequence_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "amino_acid_frequency_distribution_sequences/")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {10: 0.5, 11: 0.25, 20: 0.25},
                                                                   {"l1": {"a": 0.5, "b": 0.5}}, path / "dataset")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                          "AminoAcidFrequencyDistribution")
        params["dataset"] = dataset
        params["split_by_label"] = True
        params["result_path"] = path / "result"

        report = AminoAcidFrequencyDistribution.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        report._generate()

        self.assertTrue(os.path.isfile(path / "result/amino_acid_frequency_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/amino_acid_frequency_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/frequency_change.tsv"))
        self.assertTrue(os.path.isfile(path / "result/frequency_change.html"))

        shutil.rmtree(path)

    def test_generate_receptor_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "amino_acid_frequency_distribution_receptor/")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(100, chain_1_length_probabilities={10: 0.5, 11: 0.25,
                                                                                                      20: 0.25},
                                                                   chain_2_length_probabilities={10: 0.5, 11: 0.25,
                                                                                                 15: 0.25},
                                                                   labels={"l1": {"a": 0.5, "b": 0.5}},
                                                                   path=path / "dataset")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                          "AminoAcidFrequencyDistribution")
        params["dataset"] = dataset
        params["split_by_label"] = True
        params["result_path"] = path / "result"

        report = AminoAcidFrequencyDistribution.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        report._generate()

        self.assertTrue(os.path.isfile(path / "result/amino_acid_frequency_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/amino_acid_frequency_distribution.html"))
        self.assertTrue(os.path.isfile(path / "result/frequency_change.tsv"))
        self.assertTrue(os.path.isfile(path / "result/frequency_change.html"))

        shutil.rmtree(path)

    def test_generate_repertoire_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "amino_acid_frequency_distribution_repertoire/")

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=20,
                                                                     sequence_count_probabilities={10: 1},
                                                                     sequence_length_probabilities={10: 1},
                                                                     labels={"l1": {"a": 0.5, "b": 0.5}},
                                                                     path=path / "dataset")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                          "AminoAcidFrequencyDistribution")
        params["dataset"] = dataset
        params["relative_frequency"] = False
        params["split_by_label"] = True
        params["result_path"] = path / "result"

        report = AminoAcidFrequencyDistribution.build_object(**params)
        self.assertTrue(report.check_prerequisites())

        report._generate()

        self.assertTrue(os.path.isfile(path / "result/amino_acid_frequency_distribution.tsv"))
        self.assertTrue(os.path.isfile(path / "result/amino_acid_frequency_distribution.html"))

        df = pd.read_csv(path / "result/amino_acid_frequency_distribution.tsv", sep="\t")

        # assert that the total amino acid count at each position = n_repertoires (5) * sequences_per_repertoire (20) for each positionin the sequence (10)
        self.assertEqual([200] * 10, list(df.groupby("position")["count"].sum()))

        shutil.rmtree(path)
