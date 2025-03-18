import shutil

import numpy as np
import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.CompAIRRClusteringReport import CompAIRRClusteringReport
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.reports.ReportResult import ReportResult


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
    path = EnvironmentSettings.tmp_test_path / "compairr_clustering_report"
    PathBuilder.remove_old_and_build(path)

    try:
        # Generate random dataset
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(
            repertoire_count=3,
            sequence_count_probabilities={2: 1},
            sequence_length_probabilities={2: 1},
            labels={"disease": {True: 0.5, False: 0.5}},
            path=path
        )

        # Run report
        report = CompAIRRClusteringReport(
            dataset=dataset,
            result_path=path / "report",
            label="disease",
            compairr_path=compairr_path,
            max_distance=0,
            indels=False,
            ignore_counts=True,
            ignore_genes=True,
            threads=4,
            linkage_method='single',
            is_cdr3=True
        )

        # Generate report
        result = report._generate()

        # Test report result structure
        assert isinstance(result, ReportResult)
        assert len(result.output_figures) == 1
        assert len(result.output_tables) == 3

        # Test that files were created
        assert result.output_tables[0].path.is_file()  # distance matrix
        assert result.output_tables[1].path.is_file()  # cluster assignments
        assert result.output_figures[0].path.is_file()  # dendrogram

        # Test distance matrix format
        distance_matrix = pd.read_csv(result.output_tables[0].path, sep="\t", index_col=0)
        assert distance_matrix.shape == (3, 3)  # 3x3 matrix for 3 repertoires
        assert np.all(distance_matrix.values >= 0)  # All distances should be non-negative
        assert np.all(distance_matrix.values <= 1)  # CompAIRR normalizes distances to [0,1]
        assert np.allclose(distance_matrix.values, distance_matrix.values.T)  # Matrix should be symmetric
        assert np.all(np.diag(distance_matrix.values) == 0)  # Diagonal should be zeros

        # Test cluster assignments
        clusters = pd.read_csv(result.output_tables[2].path, sep="\t")
        assert len(clusters) == 3  # One assignment per repertoire
        assert 'repertoire_id' in clusters.columns
        assert 'cluster' in clusters.columns

    finally:
        shutil.rmtree(path)
