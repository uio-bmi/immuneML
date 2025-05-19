import logging
from typing import Tuple, List

import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.datasets.ElementDataset import ElementDataset


class TCRdistHelper:

    @staticmethod
    def get_chains(dataset: ElementDataset) -> list:
        """
        Returns the chains used in the dataset.

        Args:
            dataset: receptor dataset for which all pairwise distances between receptors will be computed

        Returns:
            a list of chains used in the dataset
        """
        return [str(Chain.get_chain(el)).lower() for el in set(dataset.data.locus.tolist())]

    @staticmethod
    def compute_tcr_dist(dataset: ElementDataset, label_names: list, cores: int = 1):
        return CacheHandler.memo_by_params((('dataset_identifier', dataset.identifier), ("type", "TCRrep")),
                                           lambda: TCRdistHelper._compute_tcr_dist(dataset, label_names, cores))

    @staticmethod
    def _compute_tcr_dist(dataset: ElementDataset, label_names: list, cores: int):
        """
        Computes the tcrdist distances by creating a TCRrep object and calling compute_distances() function.

        Parameters `ntrim` and `ctrim` in TCRrep object for CDR3 are adjusted to account for working with IMGT CDR3 definition if IMGT CDR3 was set
        as region_type for the dataset upon importing. `deduplicate` parameter is set to False as we assume that we work with clones in immuneML,
        and not individual receptors.

        Args:
            dataset: receptor dataset for which all pairwise distances between receptors will be computed
            label_names: a list of label names (e.g., specific epitopes) to be used for later classification or reports
            cores: how many cpus to use for computation

        Returns:
            an instance of TCRrep object with computed pairwise distances between all receptors in the dataset

        """
        from tcrdist.repertoire import TCRrep

        df, chains = TCRdistHelper.prepare_tcr_dist_dataframe(dataset, label_names)
        tcr_rep = TCRrep(cell_df=df, chains=chains, organism=dataset.labels["organism"], cpus=cores,
                         deduplicate=False,
                         compute_distances=False)

        tcr_rep.compute_distances()

        return tcr_rep

    @staticmethod
    def add_default_allele_to_v_gene(v_gene: str):
        if v_gene is not None and "*" not in v_gene:
            return f"{v_gene}*01"
        else:
            return v_gene

    @staticmethod
    def prepare_tcr_dist_dataframe(dataset: ElementDataset, label_names: list) -> Tuple[pd.DataFrame, List[str]]:

        df = dataset.data.topandas()

        epitope_name = 'epitope' if 'epitope' in df.columns else 'Epitope' if 'Epitope' in df.columns else ''

        if "subject" not in df:
            df['subject'] = "sub" + df['cell_id']

        df.loc[df['v_call'].str.contains("\*"), 'v_call'] = [TCRdistHelper.add_default_allele_to_v_gene(el) for el in
                                                             df.loc[df['v_call'].str.contains("\*"), 'v_call']]
        df.loc[df['j_call'].str.contains("\*"), 'j_call'] = [TCRdistHelper.add_default_allele_to_v_gene(el) for el in
                                                             df.loc[df['j_call'].str.contains("\*"), 'j_call']]
        unique_chains = [str(Chain.get_chain(el)).lower() for el in df['locus'].unique().tolist()]

        df['clone_id'] = df['cell_id' if len(unique_chains) == 2 else 'sequence_id']
        cols_to_keep = ['cdr3', 'cdr3_aa', 'v_call', 'j_call', 'duplicate_count', 'subject', epitope_name, 'clone_id']

        df_alpha, df_beta = None, None

        if 'alpha' in unique_chains:

            df_alpha = (df[df.locus == 'TRA'][cols_to_keep]
                        .rename(columns={"cdr3_aa": "cdr3_a_aa", "cdr3": "cdr3_a_nucseq", "v_call": "v_a_gene",
                                         'j_call': "j_a_gene", "duplicate_count": "count"}))
            df_alpha.loc[:, 'count'] = [1 if el in [-1, None] else el for el in df_alpha['count']]
            if len(unique_chains) == 1:
                df = df_alpha

        if 'beta' in unique_chains:

            df_beta = (df[df.locus == 'TRB'][cols_to_keep].rename(
                columns={"cdr3_aa": "cdr3_b_aa", "cdr3": "cdr3_b_nucseq", "v_call": "v_b_gene",
                         'j_call': "j_b_gene", "duplicate_count": "count"}))
            df_beta.loc[:, 'count'] = [1 if el in [-1, None] else el for el in df_beta['count']]
            if len(unique_chains) == 1:
                df = df_beta

        if len(unique_chains) == 2:
            df = df_alpha.merge(df_beta, on=['clone_id', epitope_name, 'subject', 'count'])

        return df, unique_chains
