from typing import List, Tuple

from tcrdist.summarize import _select

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class TCRdistMotifDiscovery(EncodingReport):
    """
    The report for discovering motifs in paired immune receptor data of given specificity based on TCRdist. The receptors are clustered based on the
    distance and then motifs are discovered for each cluster. The report outputs logo plots for the motifs along with the raw data used for plotting
    in csv format.

    For the implementation, `TCRdist3 <https://tcrdist3.readthedocs.io/en/latest/>`_ library was used (source code available
    `here <https://github.com/kmayerb/tcrdist3>`_). More details on the functionality used for this report are available
    `here <https://tcrdist3.readthedocs.io/en/latest/motif_gallery.html>`_.

    Original publication:
    Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
    Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_

    Arguments:

        cores (int): number of processes to use for the computation of the distance and motifs

        positive_class_name (str): the class value (e.g., epitope) used to select only the receptors that are specific to the given epitope so that
        only those sequences are used to infer motifs; the reference receptors as required by TCRdist will be the ones from the dataset that have
        different or no epitope specified in their metadata

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_tcr_dist_report: # user-defined name
            TCRdistMotifDiscovery:
                cores: 4
                positive_class_name: 'AAA' # the epitope value

    """

    @classmethod
    def build_object(cls, **kwargs):
        return TCRdistMotifDiscovery(**kwargs)

    def __init__(self, dataset: ReceptorDataset = None, result_path: str = None, name: str = None, cores: int = None, positive_class_name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.label = list(dataset.encoded_data.labels.keys())[0] if dataset is not None else None
        self.result_path = result_path
        self.cores = cores

    def generate(self) -> ReportResult:

        self.label = list(self.dataset.encoded_data.labels.keys())[0]

        from source.util.TCRdistHelper import TCRdistHelper
        from tcrdist.rep_diff import hcluster_diff

        PathBuilder.build(self.result_path)
        tcr_rep = TCRdistHelper.compute_tcr_dist(self.dataset, [self.label], self.cores)
        tcr_rep.hcluster_df, tcr_rep.Z = hcluster_diff(clone_df=tcr_rep.clone_df, pwmat=tcr_rep.pw_alpha + tcr_rep.pw_beta, x_cols=[self.label],
                                                       count_col='count')

        figures, tables = [], []

        for index, row in tcr_rep.hcluster_df.iterrows():
            figure_outputs, table_outputs = self._discover_motif_in_cluster(tcr_rep, index, row)
            figures.extend(figure_outputs)
            tables.extend(table_outputs)

        return ReportResult("TCRdist motif discovery", figures, tables)

    def _discover_motif_in_cluster(self, tcr_rep, index, row, negative_examples=None) -> Tuple[List[ReportOutput], List[ReportOutput]]:
        from tcrdist.adpt_funcs import get_centroid_seq
        from palmotif import compute_pal_motif
        from palmotif import svg_logo

        dfnode = tcr_rep.clone_df.iloc[row['neighbors_i'], ]
        figure_outputs, table_outputs = [], []

        for chain in ['a', 'b']:

            if dfnode.shape[0] > 2:
                centroid, *_ = get_centroid_seq(df=dfnode)
            else:
                centroid = dfnode[f'cdr3_{chain}_aa'].to_list()[0]

            motif, stat = compute_pal_motif(seqs=_select(df=tcr_rep.clone_df, iloc_rows=row['neighbors_i'], col=f'cdr3_{chain}_aa'),
                                            centroid=centroid, refs=negative_examples)

            figure_path = self.result_path + f"motif_{chain}_{index+1}.svg"
            svg_logo(motif, filename=figure_path)

            motif_data_path = self.result_path + f"motif_{chain}_{index+1}.csv"
            motif.to_csv(motif_data_path)

            figure_outputs.append(ReportOutput(figure_path, f'Motif {index+1} ({chain} chain)'))
            table_outputs.append(ReportOutput(motif_data_path, f'motif {index+1} ({chain} chain) csv data'))

            if negative_examples:
                stat_overview_path = self.result_path + f"motif_{chain}_{index + 1}_stat.csv"
                stat.to_csv(stat_overview_path)
                table_outputs.append(ReportOutput(stat_overview_path, f'KL divergence and log-likelihood per position given reference data: cluster '
                                                                      f'{index+1} ({chain} chain) csv data'))

        return figure_outputs, table_outputs

    def check_prerequisites(self):
        if isinstance(self.dataset, ReceptorDataset):
            return True
        else:
            return False
