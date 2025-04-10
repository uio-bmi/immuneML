import shutil
from pathlib import Path

import pytest

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.train_gen_model_reports.KLKmerComparison import KLKmerComparison
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def kl_gen_model_report(path):
    params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                      "KLKmerComparison")
    params["original_dataset"] = RandomDatasetGenerator.generate_sequence_dataset(2, {3: 1},
                                                                                  {'A': {True: 0.5, False: 0.5}},
                                                                                  path / 'gen_test_dataset1')
    params["generated_dataset"] = RandomDatasetGenerator.generate_sequence_dataset(4, {5: 1},
                                                                                   {'A': {True: 1, False: 0}},
                                                                                   path / 'gen_test_dataset2')
    params["result_path"] = path / "result"

    return KLKmerComparison.build_object(**params)


def test_report():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "kl_divergence_evaluation/")
    kl_gen_model_report(path)._generate()
    assert (path / "result" / "bad_original_sequences.html").exists()
    assert (path / "result" / "bad_simulated_sequences.html").exists()
    assert (path / "result" / "worst_true_sequences.tsv").exists()
    assert (path / "result" / "worst_simulated_sequences.tsv").exists()
    shutil.rmtree(path)
