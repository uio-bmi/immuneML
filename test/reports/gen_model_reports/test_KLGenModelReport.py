from pathlib import Path

import pytest
from immuneML.reports.train_gen_model_reports.KLGenModelReport import KLGenModelReport
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


@pytest.fixture
def dataset1():
    return RandomDatasetGenerator.generate_sequence_dataset(2, {3: 1}, {'A': {True: 0.5, False: 0.5}},
                                                            Path('tmp/gen_test_dataset1'))


@pytest.fixture
def dataset2():
    return RandomDatasetGenerator.generate_sequence_dataset(4, {5: 1}, {'A': {True: 1, False: 0}},
                                                            Path('tmp/gen_test_dataset2'))


@pytest.fixture
def kl_gen_model_report(dataset1, dataset2):
    return KLGenModelReport(dataset1, dataset2, result_path=Path('tmp/gen_test_report'))


def test_get_title(kl_gen_model_report):
    kl_gen_model_report._generate()

