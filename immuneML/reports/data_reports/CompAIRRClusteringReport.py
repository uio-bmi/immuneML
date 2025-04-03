import copy
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class CompAIRRClusteringReport(DataReport):
    """
    A report that uses CompAIRR to compute repertoire distances based on sequence overlap and performs hierarchical
    clustering on the resulting distance matrix. The clustering results are visualized using a dendrogram,
    colored by a specified label.

    The report assumes that CompAIRR (https://github.com/uio-bmi/compairr) has been installed.

    .. note::

        This is an experimental feature.

    **Specification arguments**:

    - label (str): The label by which to color the dendrogram leaves. Must be one of the metadata fields in the dataset.

    - compairr_path (str): Path to the CompAIRR executable.

    - indels (bool): Whether to allow insertions/deletions when matching sequences (default: False)

    - ignore_counts (bool): Whether to ignore sequence counts when computing overlap (default: False)

    - ignore_genes (bool): Whether to ignore V/J gene assignments when matching sequences (default: False)

    - threads (int): Number of threads to use for CompAIRR computation (default: 4)

    - linkage_method (str): The linkage method to use for hierarchical clustering (default: 'single')

    - is_cdr3 (bool): Whether the sequences represent CDR3s (default: True)

    - clustering_criterion (str): The criterion to use for clustering (default: 'distance'), as defined in
      scipy.cluster.hierarchy.linkage; valid values are 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit'

    - clustering_threshold (float): The threshold for the clustering algorithm (default: 0.5), mapped to 't' parameter
      in scipy.cluster.hierarchy.fcluster

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_compairr_clustering_report:
                    CompAIRRClusteringReport:
                        label: disease
                        compairr_path: /path/to/compairr
                        indels: false
                        ignore_counts: true
                        ignore_genes: true
                        threads: 4
                        linkage_method: single
                        is_cdr3: true
                        clustering_criterion: distance
                        clustering_threshold: 0.5

    """

    @classmethod
    def build_object(cls, **kwargs):
        valid_keys = ['label', 'compairr_path', 'indels', 'ignore_counts', 'ignore_genes', 'threads', 'linkage_method',
                      'is_cdr3', 'clustering_threshold', 'clustering_criterion', 'name']
        ParameterValidator.assert_keys(kwargs, valid_keys, CompAIRRClusteringReport.__name__,
                                       CompAIRRClusteringReport.__name__, True)
        ParameterValidator.assert_in_valid_list(kwargs['clustering_criterion'],
                                                ['distance', 'maxclust', 'monocrit', 'maxclust_monocrit'],
                                                CompAIRRClusteringReport.__name__, 'clustering_criterion')
        assert isinstance(kwargs['clustering_threshold'], (int, float)), 'clustering_threshold must be a number'

        return CompAIRRClusteringReport(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, label: str = None,
                 compairr_path: str = None, indels: bool = False, ignore_counts: bool = False,
                 ignore_genes: bool = False, threads: int = 4, linkage_method: str = 'single', is_cdr3: bool = True,
                 name: str = None, clustering_threshold: float = 0.5, clustering_criterion: str = 'distance'):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.label = label
        self.linkage_method = linkage_method
        self.compairr_path = compairr_path
        self.indels = indels
        self.ignore_counts = ignore_counts
        self.ignore_genes = ignore_genes
        self.threads = threads
        self.is_cdr3 = is_cdr3
        self.clustering_threshold = clustering_threshold
        self.clustering_criterion = clustering_criterion

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

        linkage_matrix = linkage(distance_matrix.values, method='single')
        predictions = fcluster(linkage_matrix, self.clustering_threshold, criterion=self.clustering_criterion)

        metadata = self.dataset.get_metadata([self.label, 'subject_id'], return_df=True)
        subject_ids = metadata['subject_id'].tolist()
        labels = metadata.set_index('subject_id')[self.label].to_dict()

        # Save outputs
        distance_output = self._store_matrix(distance_matrix, 'distance_matrix')
        similarity_output = self._store_matrix(similarity_matrix, 'similarity_matrix')
        similarity_heatmap = self._make_similarity_heatmap(similarity_matrix, 'similarity')
        dendrogram_output = self._create_dendrogram(subject_ids, distance_matrix, labels, linkage_matrix)
        cluster_output = self._store_cluster_assignments(self.dataset.get_example_ids(), predictions)

        return ReportResult(
            name=self.name,
            info="Hierarchical clustering of repertoires based on CompAIRR sequence overlap",
            output_figures=[dendrogram_output, similarity_heatmap],
            output_tables=[distance_output, similarity_output, cluster_output]
        )

    def compute_distance_matrix(self, similarity_matrix) -> pd.DataFrame:
        return pd.DataFrame(data=1 - similarity_matrix.values, columns=similarity_matrix.columns,
                            index=similarity_matrix.index)

    def compare_repertoires(self) -> pd.DataFrame:
        # Create similarity matrix
        n_repertoires = len(self.dataset.repertoires)
        similarity_matrix = self._init_similarity_matrix(n_repertoires)

        # Compare repertoires pairwise
        for i in range(n_repertoires):
            rep1 = self.dataset.repertoires[i]
            similarity_matrix.loc[rep1.identifier, rep1.identifier] = 1
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

        cmd_args = [str(self.compairr_path), "--matrix", str(rep1.data_filename), str(rep2.data_filename),
                    "-o", str(output_file)]
        if self.indels:
            cmd_args.append("-i")
        if self.ignore_counts:
            cmd_args.append("--ignore-counts")
        if self.ignore_genes:
            cmd_args.append("--ignore-genes")
        if self.is_cdr3:
            cmd_args.append("--cdr3")
        cmd_args.extend(["-t", str(self.threads)])
        cmd_args.extend(["-s", 'Jaccard'])

        subprocess.run(cmd_args, capture_output=True, text=True)

        if output_file.is_file():
            similarity = pd.read_csv(output_file, sep="\t").iloc[0, 1]
            similarity_matrix.loc[rep1.identifier, rep2.identifier] = similarity
            similarity_matrix.loc[rep2.identifier, rep1.identifier] = similarity

            os.remove(str(output_file))

        return similarity_matrix

    def _store_matrix(self, matrix: pd.DataFrame, name: str) -> ReportOutput:
        output_path = self.result_path / f"{name}.tsv"
        matrix.to_csv(output_path, sep="\t")
        return ReportOutput(path=output_path, name=name)

    def _make_similarity_heatmap(self, matrix: pd.DataFrame, name: str) -> ReportOutput:

        metadata = self.dataset.get_metadata(field_names=None, return_df=True)

        subject_ids = metadata['subject_id'].tolist() if 'subject_id' in metadata.columns \
            else [str(el) for el in range(matrix.shape[0])]

        mask = np.zeros_like(matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        matrix_values = copy.deepcopy(matrix.values)
        matrix_values[mask] = np.nan
        matrix_text = matrix_values.round(3).astype(str)
        matrix_text = np.where(np.isnan(matrix_values), '', matrix_text)

        fig = go.Figure(data=go.Heatmap(z=matrix_values, x=subject_ids[:-1] + [""], y=[""] + subject_ids[1:],
                                        colorscale='Darkmint',
                                        hoverongaps=False, text=matrix_text, texttemplate="%{text}"))
        fig.update_layout(template="plotly_white", title=f"{name} matrix", xaxis_title="repertoire",
                          yaxis_title='repertoire')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, autorange='reversed')
        fig.write_html(self.result_path / f"{name}_heatmap.html")

        return ReportOutput(path=self.result_path / f"{name}_heatmap.html", name=f"{name} heatmap")

    def _store_cluster_assignments(self, repertoire_ids, clusters) -> ReportOutput:
        output_path = self.result_path / "cluster_assignments.tsv"
        cluster_df_data = {
            'repertoire_id': repertoire_ids,
            'cluster': clusters
        }

        metadata = self.dataset.get_metadata(field_names=None, return_df=True)
        if 'subject_id' in metadata.columns:
            cluster_df_data['subject_id'] = metadata['subject_id'].tolist()

        cluster_df = pd.DataFrame(cluster_df_data)
        cluster_df.to_csv(output_path, sep="\t", index=False)
        return ReportOutput(
            path=output_path,
            name="Cluster Assignments",
        )

    def _create_dendrogram(self, repertoire_ids: list, distance_matrix, external_labels: dict, linkage_matrix) -> ReportOutput:
        """
        Parameters:
        -----------
        repertoire_ids : list
            List of repertoire IDs
        distance_matrix : DataFrame
            Distance matrix
        external_labels : dict
            Dictionary mapping repertoire_ids to their labels/categories
        """
        output_path = self.result_path / "dendrogram.html"

        # Create dummy data array for dendrogram
        n = distance_matrix.shape[0]
        dummy_data = np.zeros((n, 2))  # 2D dummy data

        # Initialize figure by creating upper dendrogram only
        fig = ff.create_dendrogram(dummy_data, orientation='bottom',
                                   labels=repertoire_ids, linkagefun=lambda x: linkage_matrix,
                                   distfun=lambda x: distance_matrix.values)

        # Remove trace numbers from dendrogram hover
        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'
            fig['data'][i]['hovertemplate'] = None
            fig['data'][i]['showlegend'] = False

        # Create a mapping from labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(repertoire_ids)}

        # Get the ordering of leaves from the top dendrogram and convert to indices
        dendro_leaves = [label_to_idx[label] for label in fig['layout']['xaxis']['ticktext']]
        ordered_labels = [repertoire_ids[idx] for idx in dendro_leaves]

        # Create the annotation row
        annotation_values = [external_labels[label] for label in ordered_labels]

        # Convert labels to numeric values for coloring
        unique_labels = sorted(list(set(external_labels.values())))  # sort for consistency
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        annotation_numeric = [label_to_num[label] for label in annotation_values]

        # Add annotation row
        annotation_trace = go.Heatmap(
            x=fig['layout']['xaxis']['tickvals'],
            y=['annotation'],  # Give it a name instead of [0]
            z=[annotation_numeric],
            colorscale=[[i / (len(unique_labels) - 1), color] for i, color in
                        enumerate(px.colors.qualitative.D3[:len(unique_labels)])],
            colorbar=dict(
                title=self.label,
                ticktext=unique_labels,
                tickvals=list(range(len(unique_labels))),
                tickmode='array',
                len=0.22,  # Shorter colorbar
                yanchor='top',
                y=1.0,
                xanchor='left',
                x=1.02,  # Position colorbar to the right of the plot
                thickness=20,
                bgcolor='rgba(255,255,255,0.8)',
            ),
            showscale=True,  # Set to False if you don't want the colorbar
            hovertemplate=self.label + ': %{text}<br>Repertoire: %{x}<extra></extra>',
            text=[annotation_values],
            yaxis='y3'  # Specify which axis to use
        )
        fig.add_trace(annotation_trace)

        # Reorder the distance matrix according to dendrogram leaves
        heat_data = distance_matrix.values[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]

        # Create Heatmap with custom hover template
        heatmap = [
            go.Heatmap(
                x=fig['layout']['xaxis']['tickvals'],
                y=fig['layout']['xaxis']['tickvals'],
                z=heat_data,
                colorscale='Darkmint',
                showscale=True,
                colorbar=dict(
                    title='Distance',
                    yanchor='top',
                    y=0.73,
                    xanchor='left',
                    x=1.02,  # Position closer to plot than categories legend
                    len=0.75,
                    thickness=20
                ),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Distance: %{z}<extra></extra>'
            )
        ]

        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)

        # Edit Layout with adjusted domains for the new annotation row
        fig.update_layout({
            'autosize': True,
            'showlegend': False,
            'hovermode': 'closest',
            'template': 'plotly_white',
            'xaxis': {
                'domain': [0, 1],
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'ticks': ""
            },
            'yaxis': {
                'domain': [0, 0.75],  # Reduced height for main heatmap
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'ticktext': fig['layout']['xaxis']['ticktext'],
                'tickvals': fig['layout']['xaxis']['tickvals'],
                'showticklabels': True,
                'ticks': ""
            },
            'yaxis2': {
                'domain': [0.85, 1],  # Top dendrogram
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'showticklabels': False,
                'ticks': ""
            },
            'yaxis3': {
                'domain': [0.77, 0.83],  # New annotation row
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'showticklabels': False,
                'ticks': ""
            },
        })

        fig.write_html(str(output_path))

        return ReportOutput(path=output_path, name="hierarchical_clustering_of_repertoires")
