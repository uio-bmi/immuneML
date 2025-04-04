import shutil

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.NodeDegreeDistribution import NodeDegreeDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_generate():
    compairr_runs = 0

    for compairr_path in EnvironmentSettings.compairr_paths:
        if compairr_path.exists():
            _test_generate(compairr_path)
            compairr_runs += 1
            break
        else:
            print(f"test ignored for compairr path: {compairr_path}")

    assert compairr_runs > 0


def _test_generate(compairr_path):
    path = EnvironmentSettings.tmp_test_path / "node_degree_distribution"
    PathBuilder.remove_old_and_build(path)

    dataset = RandomDatasetGenerator.generate_sequence_dataset(2, {2: 1.}, {
            "l1": {"True": 1.}, "l2": {"2": 1.}}, path)

    report = NodeDegreeDistribution(
        dataset=dataset,
        result_path=path / "report",
        compairr_path=compairr_path,
        indels=False,
        threads=4,
    )

    result = report._generate()

    assert isinstance(result, ReportResult)

    shutil.rmtree(path)
