import logging
from pathlib import Path
from typing import List, Tuple

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.TCRdistClassifier import TCRdistClassifier
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class TCRdistMotifDiscovery(MLReport):
    """
    The report for discovering motifs in paired immune receptor data of given specificity based on TCRdist3. The receptors are hierarchically
    clustered based on the tcrdist distance and then motifs are discovered for each cluster. The report outputs logo plots for the motifs along with
    the raw data used for plotting in csv format.

    For the implementation, `TCRdist3 <https://tcrdist3.readthedocs.io/en/latest/>`_ library was used (source code available
    `here <https://github.com/kmayerb/tcrdist3>`_). More details on the functionality used for this report are available
    `here <https://tcrdist3.readthedocs.io/en/latest/motif_gallery.html>`_.

    Original publications:

    Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
    Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_

    Mayer-Blackwell K, Schattgen S, Cohen-Lavi L, et al. TCR meta-clonotypes for biomarker discovery with tcrdist3: quantification of public,
    HLA-restricted TCR biomarkers of SARS-CoV-2 infection. bioRxiv. Published online December 26, 2020:2020.12.24.424260.
    `doi:10.1101/2020.12.24.424260 <https://www.biorxiv.org/content/10.1101/2020.12.24.424260v1>`_


    Arguments:

        positive_class_name (str): the class value (e.g., epitope) used to select only the receptors that are specific to the given epitope so that
        only those sequences are used to infer motifs; the reference receptors as required by TCRdist will be the ones from the dataset that have
        different or no epitope specified in their metadata; if the labels are available only on the epitope level (e.g., label is "AVFDRKSDAK" and
        classes are True and False), then here it should be specified that only the receptors with value "True" for label "AVFDRKSDAK" should be used;
        there is no default value for this argument

        cores (int): number of processes to use for the computation of the distance and motifs

        min_cluster_size (int): the minimum size of the cluster to discover the motifs for

        use_reference_sequences (bool): when showing motifs, this parameter defines if reference sequences should be provided as well as a background

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_tcr_dist_report: # user-defined name
            TCRdistMotifDiscovery:
                positive_class_name: True # class name, could also be epitope name, depending on how it's defined in the dataset
                cores: 4
                min_cluster_size: 30
                use_reference_sequences: False

    """

    @classmethod
    def build_object(cls, **kwargs):
        return TCRdistMotifDiscovery(**kwargs)

    def __init__(self, train_dataset: ReceptorDataset = None, test_dataset: ReceptorDataset = None, method: MLMethod = None, result_path: Path = None,
                 name: str = None, cores: int = None, context: dict = None, positive_class_name=None, min_cluster_size: int = None,
                 use_reference_sequences: bool = None):
        super().__init__(train_dataset, test_dataset, method, result_path, name)
        self.label = list(train_dataset.encoded_data.labels.keys())[0] if train_dataset is not None else None
        self.cores = cores
        self.positive_class_name = positive_class_name
        self.min_cluster_size = min_cluster_size
        self.use_reference_sequences = use_reference_sequences
        self.context = context

    def _generate(self) -> ReportResult:

        self.label = list(self.train_dataset.encoded_data.labels.keys())[0]

        from immuneML.util.TCRdistHelper import TCRdistHelper
        from tcrdist.rep_diff import hcluster_diff
        from tcrdist.summarize import member_summ

        PathBuilder.build(self.result_path)

        subsampled_dataset = self._extract_positive_example_dataset()
        reference_sequences = self._extract_reference_sequences()
        tcr_rep = TCRdistHelper.compute_tcr_dist(subsampled_dataset, [self.label], self.cores)
        tcr_rep.hcluster_df, tcr_rep.Z = hcluster_diff(clone_df=tcr_rep.clone_df, pwmat=tcr_rep.pw_alpha + tcr_rep.pw_beta, x_cols=["epitope"],
                                                       count_col='count')

        figures, tables = [], []

        logging.info(f'{TCRdistMotifDiscovery.__name__}: created {tcr_rep.hcluster_df.shape[0]} clusters, now discovering motifs in clusters.')

        for index, row in tcr_rep.hcluster_df.iterrows():
            if len(row['neighbors_i']) >= self.min_cluster_size:
                figure_outputs, table_outputs = self._discover_motif_in_cluster(tcr_rep, index, row, reference_sequences)
                figures.extend(figure_outputs)
                tables.extend(table_outputs)

        res_summary = member_summ(res_df=tcr_rep.hcluster_df, clone_df=tcr_rep.clone_df, addl_cols=['epitope'])
        res_summary.to_csv(self.result_path / "tcrdist_summary.csv")

        tables.append(ReportOutput(path=self.result_path / "tcrdist_summary.csv", name="TCRdist summary (csv)"))

        return ReportResult("TCRdist motif discovery", figures, tables)

    def _discover_motif_in_cluster(self, tcr_rep, index, row, negative_examples=None) -> Tuple[List[ReportOutput], List[ReportOutput]]:
        from tcrdist.adpt_funcs import get_centroid_seq
        from tcrdist.summarize import _select

        from palmotif import compute_pal_motif
        from palmotif import svg_logo

        dfnode = tcr_rep.clone_df.iloc[row['neighbors_i'],]
        figure_outputs, table_outputs = [], []

        logging.info(f"{TCRdistMotifDiscovery.__name__}: in cluster {index+1}, there are {dfnode.shape[0]} neighbors.")

        for chain in ['a', 'b']:

            if dfnode.shape[0] > 2:
                centroid, *_ = get_centroid_seq(df=dfnode)
            else:
                centroid = dfnode[f'cdr3_{chain}_aa'].to_list()[0]

            motif, stat = compute_pal_motif(seqs=_select(df=tcr_rep.clone_df, iloc_rows=row['neighbors_i'], col=f'cdr3_{chain}_aa'),
                                            centroid=centroid, refs=negative_examples[chain] if self.use_reference_sequences else None)

            figure_path = self.result_path / f"motif_{chain}_{index + 1}.svg"
            svg_logo(motif, filename=figure_path)

            motif_data_path = self.result_path / f"motif_{chain}_{index + 1}.csv"
            motif.to_csv(motif_data_path)

            figure_outputs.append(ReportOutput(figure_path, f'Motif {index + 1} ({Chain.get_chain(chain.upper()).name.lower()} chain)'))
            table_outputs.append(ReportOutput(motif_data_path, f'motif {index + 1} ({Chain.get_chain(chain.upper()).name.lower()} chain) csv data'))

        return figure_outputs, table_outputs

    def set_context(self, context: dict):
        self.context = context
        return self

    def check_prerequisites(self):
        if isinstance(self.train_dataset, ReceptorDataset) and isinstance(self.method, TCRdistClassifier):
            return True
        else:
            return False

    def _extract_positive_example_dataset(self) -> ReceptorDataset:
        positive_example_indices = []
        for index, receptor in enumerate(self.train_dataset.get_data()):
            if str(receptor.metadata[self.label]) == str(self.positive_class_name):
                positive_example_indices.append(index)

        subsampled_dataset = self.train_dataset.make_subset(example_indices=positive_example_indices, path=self.result_path,
                                                            dataset_type=ReceptorDataset.SUBSAMPLED)

        logging.info(f"{TCRdistMotifDiscovery.__name__}: extracted only positive examples from the training dataset (examples with class = "
                     f"{self.positive_class_name}) for motif discovery. Example count in the new dataset: {subsampled_dataset.get_example_count()}.")

        return subsampled_dataset

    def _extract_reference_sequences(self) -> dict:

        reference_sequences = {'a': [], 'b': []}

        if self.use_reference_sequences:
            for index, receptor in enumerate(self.train_dataset.get_data()):
                if str(receptor.metadata[self.label]) != str(self.positive_class_name):
                    reference_sequences['a'].append(receptor.alpha.amino_acid_sequence)
                    reference_sequences['b'].append(receptor.beta.amino_acid_sequence)

        return reference_sequences
