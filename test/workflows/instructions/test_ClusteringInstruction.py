from pathlib import Path
import unittest
import shutil
import os

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.reports.ml_reports.ClusteringReport import ClusteringReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringInstruction
from immuneML.workflows.instructions.clustering.ClusteringUnit import ClusteringUnit

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class TestClusteringInstruction(unittest.TestCase):

    def setUp(self) -> None:
        os.environ["CACHE_TYPE"] = "TEST"

    def create_dataset(self, path: Path) -> RepertoireDataset:
        repertoires = [Repertoire.build(sequence_aas=[["AA"], ["AC"], ["CC"], ["AAA"], ["AACC"], ["AAAC"]], sequence_identifiers=["1", "2", "3", "4", "5", "6"], path=path / "rep1"),
                       Repertoire.build(sequence_aas=[["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], sequence_identifiers=["7", "8", "9", "10", "11", "12"], path=path / "rep2")]

        dataset = RepertoireDataset(repertoires=repertoires)
        return dataset

    def test_run(self):
        path = Path("tmp_test_clustering_instruction/")
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        clustering_units = {
            "clustering_analysis_1": ClusteringUnit(dataset=dataset, encoder=Word2VecEncoder(k=2, vector_size=10),
                                                    clustering_method=KMeans(n_clusters=2),
                                                    report=ClusteringReport(),
                                                    number_of_processes=2),
            "clustering_analysis_2": ClusteringUnit(dataset=dataset, encoder=Word2VecEncoder(k=2, vector_size=10),
                                                    clustering_method=KMeans(n_clusters=2),
                                                    dimensionality_reduction=PCA(n_components=2),
                                                    report=ClusteringReport(),
                                                    number_of_processes=2)
        }

        instruction = ClusteringInstruction(clustering_units=clustering_units, name="clustering")
        instruction.run(path / "results/")

        self.assertTrue(os.path.isfile(path / "results/clustering/analysis_clustering_analysis_1/report/cluster_plot.html"))
        self.assertTrue(os.path.isfile(path / "results/clustering/analysis_clustering_analysis_2/report/cluster_plot.html"))

        shutil.rmtree(path)
