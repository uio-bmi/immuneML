import shutil
from pathlib import Path

import pytest

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.train_gen_model_reports.KLGenModelReport import KLGenModelReport
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


@pytest.fixture
def dataset1():
    return RandomDatasetGenerator.generate_sequence_dataset(2, {3: 1}, {'A': {True: 0.5, False: 0.5}},
                                                            Path('tmp/gen_test_dataset1'))


@pytest.fixture
def dataset2():
    return RandomDatasetGenerator.generate_sequence_dataset(4, {5: 1}, {'A': {True: 1, False: 0}},
                                                            Path('tmp/gen_test_dataset2'))

@pytest.fixture
def path():
    return PathBuilder.build(EnvironmentSettings.tmp_test_path / "kl_divergence_evaluation/")

@pytest.fixture
def kl_gen_model_report(dataset1, dataset2, path):
    params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                      "KLGenModelReport")
    params["original_dataset"] = dataset1
    params["generated_dataset"] = dataset2
    params["result_path"] = path / "result"

    return KLGenModelReport.build_object(**params)


def test_get_title(kl_gen_model_report, path):
    kl_gen_model_report._generate()
    shutil.rmtree(path)