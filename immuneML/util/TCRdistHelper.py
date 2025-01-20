import logging

import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.SequenceParams import RegionType


class TCRdistHelper:

    @staticmethod
    def compute_tcr_dist(dataset: ReceptorDataset, label_names: list, cores: int = 1):
        return CacheHandler.memo_by_params((('dataset_identifier', dataset.identifier), ("type", "TCRrep")),
                                           lambda: TCRdistHelper._compute_tcr_dist(dataset, label_names, cores))

    @staticmethod
    def _compute_tcr_dist(dataset: ReceptorDataset, label_names: list, cores: int):
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

        df = TCRdistHelper.prepare_tcr_dist_dataframe(dataset, label_names)
        tcr_rep = TCRrep(cell_df=df, chains=['alpha', 'beta'], organism=dataset.labels["organism"], cpus=cores,
                         deduplicate=False,
                         compute_distances=False)

        if 'region_type' not in dataset.labels:
            logging.warning(
                f"{TCRdistHelper.__name__}: Parameter 'region_type' was not set for dataset {dataset.name}, keeping default tcrdist "
                f"values for parameters 'ntrim' and 'ctrim'. For more information, see tcrdist3 documentation. To avoid this warning, "
                f"set the region type when importing the dataset.")
        elif dataset.labels['region_type'] == RegionType.IMGT_CDR3:
            tcr_rep.kargs_a['cdr3_a_aa']['ntrim'] = 2
            tcr_rep.kargs_a['cdr3_a_aa']['ctrim'] = 1
            tcr_rep.kargs_b['cdr3_b_aa']['ntrim'] = 2
            tcr_rep.kargs_b['cdr3_b_aa']['ctrim'] = 1
        elif dataset.labels['region_type'] != RegionType.IMGT_JUNCTION:
            raise RuntimeError(
                f"{TCRdistHelper.__name__}: TCRdist metric can be computed only if IMGT_CDR3 or IMGT_JUNCTION are used as region "
                f"types, but for dataset {dataset.name}, it is set to {dataset.labels['region_type']} instead.")

        tcr_rep.compute_distances()

        return tcr_rep

    @staticmethod
    def add_default_allele_to_v_gene(v_gene: str):
        if v_gene is not None and "*" not in v_gene:
            return f"{v_gene}*01"
        else:
            return v_gene

    @staticmethod
    def prepare_tcr_dist_dataframe(dataset: ReceptorDataset, label_names: list) -> pd.DataFrame:
        if len(label_names) > 1:
            raise NotImplementedError(
                f"TCRdist: multiple labels specified ({str(label_names)[1:-1]}), but only single label binary class "
                f"is currently supported in immuneML.")

        df = dataset.data.topandas()

        epitope_name = 'epitope' if 'epitope' in df.columns else 'Epitope' if 'Epitope' in df.columns else ''

        if "subject" not in df:
            df['subject'] = "sub" + df['cell_id']

        df.loc[df['v_call'].str.contains("\*"), 'v_call'] = [TCRdistHelper.add_default_allele_to_v_gene(el) for el in
                                                             df.loc[df['v_call'].str.contains("\*"), 'v_call']]
        df.loc[df['j_call'].str.contains("\*"), 'j_call'] = [TCRdistHelper.add_default_allele_to_v_gene(el) for el in
                                                             df.loc[df['j_call'].str.contains("\*"), 'j_call']]

        df['clone_id'] = df['cell_id']
        cols_to_keep = ['cdr3', 'cdr3_aa', 'v_call', 'j_call', 'duplicate_count', 'subject', epitope_name, 'clone_id']

        df_alpha = (df[df.locus == 'TRA'][cols_to_keep]
                    .rename(columns={"cdr3_aa": "cdr3_a_aa", "cdr3": "cdr3_a_nucseq", "v_call": "v_a_gene",
                                     'j_call': "j_a_gene", "duplicate_count": "count"}))
        df_alpha.loc[:, 'count'] = [1 if el in [-1, None] else el for el in df_alpha['count']]

        df_beta = (df[df.locus == 'TRB'][cols_to_keep].rename(
            columns={"cdr3_aa": "cdr3_b_aa", "cdr3": "cdr3_b_nucseq", "v_call": "v_b_gene",
                     'j_call': "j_b_gene", "duplicate_count": "count"}))
        df_beta.loc[:, 'count'] = [1 if el in [-1, None] else el for el in df_beta['count']]

        df = df_alpha.merge(df_beta, on=['clone_id', epitope_name, 'subject', 'count'])

        return df
