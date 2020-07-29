import logging
from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance
from tcrdist import plotting
from tcrdist.mappers import populate_legacy_fields
from tcrdist.subset import TCRsubset

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder
from source.util.TCRdistHelper import TCRdistHelper


class TCRdistMotifDiscovery(EncodingReport):
    """
    The report for discovering motifs in paired immune receptor data of given specificity based on TCRdist. The receptors are clustered based on the
    distance and then motifs are discovered for each cluster. The report outputs logo plots for the motifs along with the raw data used for plotting
    in csv format.

    For the implementation, `TCRdist2 <https://tcrdist2.readthedocs.io/en/latest/index.html>`_ library was used. More details on the functionality used
    for this report are available `here <https://tcrdist2.readthedocs.io/en/latest/HotStart.html#discover-motifs>`_.

    Original publication:
    Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
    Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_

    Arguments:

        cores (int): number of processes to use for the computation of the distance and motifs

        max_cluster_count (int): max number of clusters to be extracted for motif discovery

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_tcr_dist_report: # user-defined name
            TCRdistMotifDiscovery:
                cores: 4
                max_cluster_count: 20

    """

    @classmethod
    def build_object(cls, **kwargs):
        return TCRdistMotifDiscovery(**kwargs)

    def __init__(self, dataset: ReceptorDataset, result_path: str = None, name: str = None, cores: int = None, max_cluster_count: int = None):
        super().__init__(name)
        self.dataset = dataset
        self.label = list(dataset.encoded_data.labels.keys())[0]
        self.result_path = result_path
        self.cores = cores
        self.max_cluster_count = max_cluster_count

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        tcr_rep = TCRdistHelper.compute_tcr_dist(self.dataset, [self.label], self.cores)
        compressed_dmat = distance.squareform(tcr_rep.paired_tcrdist, force="vector")
        Z = linkage(compressed_dmat, method="complete")
        # den = dendrogram(Z, color_threshold=np.inf, no_plot=True)
        cluster_index = fcluster(Z, t=self.max_cluster_count, criterion="maxclust")
        tcr_rep.clone_df['cluster_index'] = cluster_index

        cluster_indices = np.unique(cluster_index)
        figures, tables = [], []

        for value in cluster_indices:
            figure_outputs, table_outputs = self._discover_motif_in_cluster(cluster_index, value, tcr_rep)
            figures.extend(figure_outputs)
            tables.extend(table_outputs)

        return ReportResult("TCRdist motif discovery", figures, tables)

    def _discover_motif_in_cluster(self, cluster_indices, cluster_index, tcr_rep) -> Tuple[List[ReportOutput], List[ReportOutput]]:
        logging.info(f"TCRdistMotifDiscovery: processing cluster {cluster_index} of {max(cluster_indices)}...")
        criteria = (cluster_indices == cluster_index)
        clone_df_subset = tcr_rep.clone_df[criteria]
        epitopes = np.unique(self.dataset.encoded_data.labels[self.label])
        figure_outputs, table_outputs = [], []

        for epitope in epitopes:
            logging.info(f"TCRdistMotifDiscovery: processing epitope {epitope}...")
            figures, tables = self._discover_motif_for_epitope(clone_df_subset, epitope, tcr_rep, epitopes, cluster_index)
            figure_outputs.extend(figures)
            table_outputs.extend(tables)

        return figure_outputs, table_outputs

    def _discover_motif_for_epitope(self, clone_df_subset, epitope, tcr_rep, epitopes, cluster_index):
        clone_df_subset = clone_df_subset[clone_df_subset.epitope == epitope].copy()
        dist_a_subset = tcr_rep.dist_a.loc[clone_df_subset.clone_id, clone_df_subset.clone_id].copy()
        dist_b_subset = tcr_rep.dist_b.loc[clone_df_subset.clone_id, clone_df_subset.clone_id].copy()

        clone_df_subset = populate_legacy_fields(df=clone_df_subset, chains=['alpha', 'beta'])

        ts = TCRsubset(clone_df_subset, organism=self.dataset.params['organism'], epitopes=epitopes, epitope=epitope, chains=["A", "B"],
                       dist_a=dist_a_subset, dist_b=dist_b_subset)

        motif_df = ts.find_motif()
        motif_path = f"{self.result_path}motif_{cluster_index + 1}_{epitope}.csv"
        motif_df.to_csv(motif_path, index=False)
        figure_outputs = self._plot_motifs_per_chain(motif_df, ts, cluster_index, epitope)
        table_outputs = [ReportOutput(motif_path, f"motif {cluster_index + 1} - csv data")]
        return figure_outputs, table_outputs

    def _plot_motifs_per_chain(self, motif_df, ts: TCRsubset, cluster_index, epitope) -> List[ReportOutput]:
        figure_outputs = []
        for i, row in motif_df[motif_df.ab == "A"].iterrows():
            figure_outputs.append(self._plot_motif(ts, row, cluster_index, i, "alpha", epitope))
        for i, row in ts.motif_df[ts.motif_df.ab == "B"].iterrows():
            figure_outputs.append(self._plot_motif(ts, row, cluster_index, i, "beta", epitope))

        return figure_outputs

    def _plot_motif(self, ts, row, cluster_index, i, chain_name, epitope):
        StoreIOMotif_instance = ts.eval_motif(row)
        path = f"{self.result_path}motif_{cluster_index+1}_{i+1}_{chain_name}_{epitope}.svg"
        plotting.plot_pwm(StoreIOMotif_instance, create_file=True, my_height=200, my_width=600,
                          output=path)
        return ReportOutput(path, f"motif {cluster_index + 1}-{i + 1} (epitope: {epitope}): {chain_name} chain")

    def check_prerequisites(self):
        if isinstance(self.dataset, ReceptorDataset):
            return True
        else:
            return False
