import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class CompAIRRClusteringReport(DataReport):
    """
    A report that uses CompAIRR to compute repertoire distances based on sequence overlap and performs hierarchical
    clustering on the resulting distance matrix. The clustering results are visualized using a dendrogram,
    colored by a specified label.

    The report assumes that CompAIRR (https://github.com/uio-bmi/compairr) has been installed.

    .. note::

        This is an experimental feature.

    Arguments:

    - label (str): The label by which to color the dendrogram leaves. Must be one of the metadata fields in the dataset.

    - compairr_path (str): Path to the CompAIRR executable.

    - max_distance (int): Max Hamming distance allowed between sequences (default: 1)

    - indels (bool): Whether to allow insertions/deletions when matching sequences (default: False)

    - ignore_counts (bool): Whether to ignore sequence counts when computing overlap (default: False)

    - ignore_genes (bool): Whether to ignore V/J gene assignments when matching sequences (default: False)

    - threads (int): Number of threads to use for CompAIRR computation (default: 4)

    - linkage_method (str): The linkage method to use for hierarchical clustering (default: 'single')

    - is_cdr3 (bool): Whether the sequences represent CDR3s (default: True)

    - clustering_threshold (float): The threshold to use for cutting the dendrogram into clusters (default: 0.5)

    """

    @classmethod
    def build_object(cls, **kwargs):
        return CompAIRRClusteringReport(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, label: str = None,
                 compairr_path: str = None,
                 max_distance: int = 1, indels: bool = False, ignore_counts: bool = False, ignore_genes: bool = False,
                 threads: int = 4, linkage_method: str = 'single', is_cdr3: bool = True,
                 clustering_threshold: float = 0.5, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.label = label
        self.linkage_method = linkage_method
        self.compairr_path = compairr_path
        self.max_distance = max_distance
        self.indels = indels
        self.ignore_counts = ignore_counts
        self.ignore_genes = ignore_genes
        self.threads = threads
        self.is_cdr3 = is_cdr3
        self.clustering_threshold = clustering_threshold

    def check_prerequisites(self) -> bool:
        if not self.compairr_path:
            logging.warning("CompAIRR path not provided. CompAIRR must be installed and available in the system PATH.")
            return False
        if not isinstance(self.dataset, RepertoireDataset):
            logging.warning("CompAIRRClusteringReport requires a RepertoireDataset as input.")
            return False
        return True

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self.check_label()

        similarity_matrix = self.compare_repertoires()
        distance_matrix = self.compute_distance_matrix(similarity_matrix)

        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=self.linkage_method,
                                        metric='precomputed')
        predictions = model.fit_predict(distance_matrix.values)

        # Save outputs
        distance_output = self._store_matrix(distance_matrix, 'distance_matrix')
        similarity_output = self._store_matrix(similarity_matrix, 'similarity_matrix')
        dendrogram_output = self._create_dendrogram(model, self.dataset.get_metadata(['subject_id'])['subject_id'])
        cluster_output = self._store_cluster_assignments(self.dataset.get_example_ids(), predictions)

        return ReportResult(
            name=self.name,
            info="Hierarchical clustering of repertoires based on CompAIRR sequence overlap",
            output_figures=[dendrogram_output],
            output_tables=[distance_output, similarity_output, cluster_output]
        )

    def compute_distance_matrix(self, similarity_matrix) -> pd.DataFrame:
        return pd.DataFrame(data=1 - similarity_matrix.values / similarity_matrix.values.max(),
                            columns=similarity_matrix.columns, index=similarity_matrix.index)

    def compare_repertoires(self) -> pd.DataFrame:
        # Create similarity matrix
        n_repertoires = len(self.dataset.repertoires)
        similarity_matrix = self._init_similarity_matrix(n_repertoires)

        # Compare repertoires pairwise
        for i in range(n_repertoires):
            rep1 = self.dataset.repertoires[i]
            similarity_matrix.loc[rep1.identifier, rep1.identifier] = rep1.get_element_count()
            for j in range(i + 1, n_repertoires):
                rep2 = self.dataset.repertoires[j]
                similarity_matrix = self._run_compairr(rep1, rep2, similarity_matrix)

        return similarity_matrix

    def check_label(self):
        if self.label not in self.dataset.get_label_names():
            raise ValueError(
                f"Label '{self.label}' not found in dataset metadata. Available labels: {self.dataset.get_label_names()}")

    def _init_similarity_matrix(self, n_repertoires: int) -> pd.DataFrame:
        return pd.DataFrame(
            np.zeros((n_repertoires, n_repertoires)),
            index=[rep.identifier for rep in self.dataset.repertoires],
            columns=[rep.identifier for rep in self.dataset.repertoires]
        )

    def _run_compairr(self, rep1: Repertoire, rep2: Repertoire, similarity_matrix: pd.DataFrame):
        output_file = self.result_path / f"overlap_{rep1.identifier}_{rep2.identifier}.tsv"

        cmd_args = [str(self.compairr_path), "-m", str(rep1.data_filename), str(rep2.data_filename),
                    "-d", str(self.max_distance), "-o", str(output_file)]
        if self.indels:
            cmd_args.append("-i")
        if self.ignore_counts:
            cmd_args.append("-f")
        if self.ignore_genes:
            cmd_args.append("-g")
        if self.is_cdr3:
            cmd_args.append("--cdr3")

        subprocess.run(cmd_args, capture_output=True, text=True)

        # Calculate similarity as sum of matches
        if output_file.is_file():
            similarity = pd.read_csv(output_file, sep="\t").sum().sum()
            similarity_matrix.loc[rep1.identifier, rep2.identifier] = similarity
            similarity_matrix.loc[rep2.identifier, rep1.identifier] = similarity

            os.remove(str(output_file))

        return similarity_matrix

    def _store_matrix(self, matrix: pd.DataFrame, name: str) -> ReportOutput:
        output_path = self.result_path / f"{name}.tsv"
        matrix.to_csv(output_path, sep="\t")
        return ReportOutput(
            path=output_path,
            name=name,
        )

    def _store_cluster_assignments(self, repertoire_ids, clusters) -> ReportOutput:
        output_path = self.result_path / "cluster_assignments.tsv"
        cluster_df = pd.DataFrame({
            'repertoire_id': repertoire_ids,
            'cluster': clusters
        })
        cluster_df.to_csv(output_path, sep="\t", index=False)
        return ReportOutput(
            path=output_path,
            name="Cluster Assignments",
        )

    def _create_dendrogram(self, model, repertoire_ids: list) -> ReportOutput:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(len(repertoire_ids) * 0.3 + 1, 10))

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        labels = self.dataset.get_metadata([self.label])[self.label]
        import plotly.express as px
        unique_label_values = list(self.dataset.labels[self.label])
        colors = {rep_id: tuple([round(float(el) / 255, 3) for el in
                                 px.colors.qualitative.Prism[unique_label_values.index(labels[ind])][4:-1].split(', ')])
                  for ind, rep_id in enumerate(repertoire_ids)}

        ddata = dendrogram(linkage_matrix, labels=np.array(repertoire_ids))

        for leaf, leaf_color in zip(plt.gca().get_xticklabels(), ddata["leaves_color_list"]):
            leaf.set_color(colors[leaf.get_text()])

        plt.xticks(rotation=90)

        output_path = self.result_path / "hierarchical_clustering_of_repertoires.png"
        plt.savefig(str(output_path), dpi=300)

        return ReportOutput(path=output_path, name="hierarchical_clustering_of_repertoires")
