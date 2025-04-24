import shutil

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.NodeDegreeDistribution import NodeDegreeDistribution
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


@pytest.fixture
def dummy_sequence_dataset(tmp_path):
    dataset = MagicMock(spec=SequenceDataset)
    dataset.filename = tmp_path / "dummy_dataset.tsv"
    return dataset


@pytest.fixture
def dummy_repertoire_dataset(tmp_path):
    dataset = MagicMock(spec=RepertoireDataset)
    dataset.filename = tmp_path / "dummy_repertoire_dataset.tsv"
    return dataset


@pytest.fixture
def report(tmp_path, dummy_sequence_dataset):
    return NodeDegreeDistribution(
        dataset=dummy_sequence_dataset,
        result_path=tmp_path,
        compairr_path="/path/to/compairr",
        region_type=RegionType.IMGT_CDR3,
        indels=False,
        ignore_genes=False,
        hamming_distance=1,
        threads=2,
        name="test_report"
    )


@pytest.mark.parametrize("region_type, compairr_path, dataset_type, expected", [
    (RegionType.IMGT_CDR3, "/path/to/compairr", "sequence", True),
    (RegionType.IMGT_CDR3, "/path/to/compairr", "repertoire", True),
    (RegionType.IMGT_FR1, "/path/to/compairr", "sequence", False),
    (RegionType.IMGT_CDR3, None, "sequence", False),
])
def test_check_prerequisites_all(
        report, dummy_sequence_dataset, dummy_repertoire_dataset,
        region_type, compairr_path, dataset_type, expected
):
    report.region_type = region_type
    report.compairr_path = compairr_path
    report.dataset = dummy_sequence_dataset if dataset_type == "sequence" else dummy_repertoire_dataset
    assert report.check_prerequisites() is expected


def test_build_object_sets_enum_correctly():
    obj = NodeDegreeDistribution.build_object(
        dataset=MagicMock(),
        result_path=Path("/tmp/fake"),
        compairr_path="/usr/bin/compairr",
        region_type="IMGT_CDR3"
    )
    assert obj.region_type == RegionType.IMGT_CDR3


def test_generate():
    compairr_runs = 0

    for compairr_path in EnvironmentSettings.compairr_paths:
        if compairr_path.exists():
            assert_report_outputs_sequence_dataset(compairr_path)
            assert_report_outputs_repertoire_dataset(compairr_path)
            compairr_runs += 1
            break
        else:
            print(f"test ignored for compairr path: {compairr_path}")

    assert compairr_runs > 0


def assert_report_outputs_sequence_dataset(compairr_path):
    path = EnvironmentSettings.tmp_test_path / "node_degree_distribution"
    PathBuilder.remove_old_and_build(path)

    dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {3: 1.},
                                                               {"l1": {'True': 0.5, 'False': 0.5}}, path,
                                                               region_type="IMGT_JUNCTION")

    report = NodeDegreeDistribution(
        dataset=dataset,
        result_path=path / "node_degree_distribution",
        compairr_path=compairr_path,
        indels=False,
        ignore_genes=False,
        region_type=RegionType.IMGT_JUNCTION,
        hamming_distance=2,
        threads=4,
    )

    result = report._generate()

    assert isinstance(result, ReportResult)
    assert len(result.output_figures) == 1
    assert len(result.output_tables) == 1

    assert result.output_tables[0].path.is_file()  # node degree distribution
    assert result.output_figures[0].path.is_file()  # node degree distribution histogram

    shutil.rmtree(path)


def assert_report_outputs_repertoire_dataset(compairr_path):
    path = EnvironmentSettings.tmp_test_path / "node_degree_distribution"
    PathBuilder.remove_old_and_build(path)

    repertoire_count = 3
    labels = {"l1": {'True': 1}}
    labels_size = sum(len(inner) for inner in labels.values())
    dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count, {100: 1},
                                                                 {3: 1}, labels, path)

    report = NodeDegreeDistribution(
        dataset=dataset,
        result_path=path / "node_degree_distribution",
        compairr_path=compairr_path,
        indels=False,
        ignore_genes=False,
        region_type=RegionType.IMGT_CDR3,
        hamming_distance=2,
        per_repertoire=True,
        per_label=True,
        threads=4
    )

    result = report._generate()

    assert isinstance(result, ReportResult)
    assert len(result.output_figures) == repertoire_count + labels_size + 1
    assert len(result.output_tables) == repertoire_count + labels_size + 1

    for i in range(repertoire_count + 1):
        assert result.output_tables[i].path.is_file()  # node degree distribution
        assert result.output_figures[i].path.is_file()  # node degree distribution histogram

    shutil.rmtree(path)
