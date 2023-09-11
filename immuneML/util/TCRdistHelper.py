import logging

import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.RegionType import RegionType


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
        tcr_rep = TCRrep(cell_df=df, chains=['alpha', 'beta'], organism=dataset.labels["organism"], cpus=cores, deduplicate=False,
                         compute_distances=False)

        if 'region_type' not in dataset.labels:
            logging.warning(f"{TCRdistHelper.__name__}: Parameter 'region_type' was not set for dataset {dataset.name}, keeping default tcrdist "
                            f"values for parameters 'ntrim' and 'ctrim'. For more information, see tcrdist3 documentation. To avoid this warning, "
                            f"set the region type when importing the dataset.")
        elif dataset.labels['region_type'] == RegionType.IMGT_CDR3:
            tcr_rep.kargs_a['cdr3_a_aa']['ntrim'] = 2
            tcr_rep.kargs_a['cdr3_a_aa']['ctrim'] = 1
            tcr_rep.kargs_b['cdr3_b_aa']['ntrim'] = 2
            tcr_rep.kargs_b['cdr3_b_aa']['ctrim'] = 1
        elif dataset.labels['region_type'] != RegionType.IMGT_JUNCTION:
            raise RuntimeError(f"{TCRdistHelper.__name__}: TCRdist metric can be computed only if IMGT_CDR3 or IMGT_JUNCTION are used as region "
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
            raise NotImplementedError(f"TCRdist: multiple labels specified ({str(label_names)[1:-1]}), but only single label binary class "
                                      f"is currently supported in immuneML.")
        label_name = label_names[0]

        subject, epitope, count, v_a_gene, j_a_gene, cdr3_a_aa, v_b_gene, j_b_gene, cdr3_b_aa, clone_id, cdr3_b_nucseq, cdr3_a_nucseq = \
            [], [], [], [], [], [], [], [], [], [], [], []

        for receptor in dataset.get_data():
            subject.append(receptor.metadata["subject"] if "subject" in receptor.metadata else "sub" + receptor.identifier)
            epitope.append(receptor.metadata[label_name])
            count.append(receptor.get_chain("alpha").metadata.duplicate_count
                         if receptor.get_chain("alpha").metadata.duplicate_count == receptor.get_chain("beta").metadata.duplicate_count
                            and receptor.get_chain("beta").metadata.duplicate_count is not None else 1)
            v_a_gene.append(TCRdistHelper.add_default_allele_to_v_gene(receptor.get_chain('alpha').metadata.v_call))
            j_a_gene.append(receptor.get_chain('alpha').metadata.j_call)
            cdr3_a_aa.append(receptor.get_chain('alpha').sequence_aa)
            cdr3_a_nucseq.append(receptor.get_chain("alpha").sequence)
            v_b_gene.append(TCRdistHelper.add_default_allele_to_v_gene(receptor.get_chain('beta').metadata.v_call))
            j_b_gene.append(receptor.get_chain('beta').metadata.v_call)
            cdr3_b_aa.append(receptor.get_chain('beta').sequence_aa)
            cdr3_b_nucseq.append(receptor.get_chain("beta").sequence)
            clone_id.append(receptor.identifier)

        if all(item is not None for item in cdr3_a_nucseq) and all(item is not None for item in cdr3_b_nucseq):
            return pd.DataFrame({"subject": subject, "epitope": epitope, "count": count, "v_a_gene": v_a_gene, "j_a_gene": j_a_gene,
                                 "cdr3_a_aa": cdr3_a_aa, "v_b_gene": v_b_gene, "j_b_gene": j_b_gene, "cdr3_b_aa": cdr3_b_aa, "clone_id": clone_id,
                                 "cdr3_b_nucseq": cdr3_b_nucseq, "cdr3_a_nucseq": cdr3_a_nucseq})
        else:
            return pd.DataFrame({"subject": subject, "epitope": epitope, "count": count, "v_a_gene": v_a_gene, "j_a_gene": j_a_gene,
                                 "cdr3_a_aa": cdr3_a_aa, "v_b_gene": v_b_gene, "j_b_gene": j_b_gene, "cdr3_b_aa": cdr3_b_aa, "clone_id": clone_id})
