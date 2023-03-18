import logging

import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.receptor.RegionType import RegionType


class TCRdistHelper:

    @staticmethod
    def compute_tcr_dist(dataset, label_names: list, cores: int = 1, chains=list):
        if chains is None:
            chains = ["beta"]
        return CacheHandler.memo_by_params((('dataset_identifier', dataset.identifier), ("type", "TCRrep")),
                                           lambda: TCRdistHelper._compute_tcr_dist(dataset, label_names, cores, chains))

    @staticmethod
    def _compute_tcr_dist(dataset, label_names: list, cores: int, chains: []):
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
        TCRdistHelper.check_label_names(label_names)
        label_name = label_names[0]

        df = TCRdistHelper.prepare_tcr_dist_dataframe(dataset, label_name, chains)

        tcr_rep = TCRdistHelper.create_tcr_rep(df, dataset, cores, chains)

        tcr_rep.compute_distances()

        return tcr_rep

    @staticmethod
    def check_label_names(label_names):
        if len(label_names) > 1:
            raise NotImplementedError(f"TCRdist: multiple labels specified ({str(label_names)[1:-1]}), but only single label binary class "
                                      f"is currently supported in immuneML.")
        elif len(label_names) < 1:
            label_names.append('epitope')

    @staticmethod
    def create_tcr_rep(df, dataset, cores, chains):
        from tcrdist.repertoire import TCRrep

        tcr_rep = TCRrep(cell_df=df, chains=chains, organism=dataset.labels["organism"], cpus=cores, deduplicate=False,
                         compute_distances=False)

        if 'region_type' not in dataset.labels:
            logging.warning(f"{TCRdistHelper.__name__}: Parameter 'region_type' was not set for dataset {dataset.name}, keeping default tcrdist "
                            f"values for parameters 'ntrim' and 'ctrim'. For more information, see tcrdist3 documentation. To avoid this warning, "
                            f"set the region type when importing the dataset.")
        elif dataset.labels['region_type'] == RegionType.IMGT_CDR3:
            TCRdistHelper.adjust_trim_parameters(tcr_rep, chains)
        elif dataset.labels['region_type'] != RegionType.IMGT_JUNCTION:
            raise RuntimeError(f"{TCRdistHelper.__name__}: TCRdist metric can be computed only if IMGT_CDR3 or IMGT_JUNCTION are used as region "
                               f"types, but for dataset {dataset.name}, it is set to {dataset.labels['region_type']} instead.")

        return tcr_rep

    @staticmethod
    def adjust_trim_parameters(tcr_rep, chains):
        if "alpha" in chains:
            tcr_rep.kargs_a['cdr3_a_aa']['ntrim'] = 2
            tcr_rep.kargs_a['cdr3_a_aa']['ctrim'] = 1
        if "beta" in chains:
            tcr_rep.kargs_b['cdr3_b_aa']['ntrim'] = 2
            tcr_rep.kargs_b['cdr3_b_aa']['ctrim'] = 1

    @staticmethod
    def add_default_allele_to_v_gene(v_gene: str):
        if v_gene is not None and "*" not in v_gene:
            return f"{v_gene}*01"
        else:
            return v_gene

    @staticmethod
    def prepare_tcr_dist_dataframe(dataset, label_name: str, chains: list) -> pd.DataFrame:
        fields = TCRdistHelper.prepare_tcr_dist_fields(chains)

        for receptor in dataset.get_data():
            TCRdistHelper.process_receptor(receptor, label_name, fields, chains)

        if "alpha" in chains and all(item is None or item == "" for item in fields["cdr3_a_nucseq"]):
            fields.pop("cdr3_a_nucseq")
        if "beta" in chains and all(item is None or item == "" for item in fields["cdr3_b_nucseq"]):
            fields.pop("cdr3_b_nucseq")

        return pd.DataFrame(fields)

    @staticmethod
    def process_receptor(receptor, label_name, fields, chains):
        fields["subject"].append(receptor.metadata["subject"] if "subject" in receptor.metadata else "sub" + receptor.identifier)
        fields["epitope"].append(receptor.metadata[label_name])

        TCRdistHelper.process_receptor_count(receptor, fields, chains)

        TCRdistHelper.process_receptor_alpha_chain(receptor, fields, chains)
        TCRdistHelper.process_receptor_beta_chain(receptor, fields, chains)

        fields["clone_id"].append(receptor.identifier)

    @staticmethod
    def process_receptor_count(receptor, fields, chains):
        if type(receptor).__name__ == "TCABReceptor":
            fields["count"].append(receptor.get_chain("alpha").metadata.count
                                   if receptor.get_chain("alpha").metadata.count == receptor.get_chain("beta").metadata.count
                                      and receptor.get_chain("beta").metadata.count is not None else 1)
        elif type(receptor).__name__ == "ReceptorSequence":
            if receptor.metadata.chain.name.lower() in chains:
                fields["count"].append(receptor.metadata.count if receptor.metadata.count is not None else 1)

    @staticmethod
    def process_receptor_alpha_chain(receptor, fields, chains):
        if "alpha" in chains:
            if type(receptor).__name__ == "TCABReceptor":
                fields["v_a_gene"].append(TCRdistHelper.add_default_allele_to_v_gene(receptor.get_chain('alpha').metadata.v_allele))
                fields["j_a_gene"].append(receptor.get_chain('alpha').metadata.j_allele)
                fields["cdr3_a_aa"].append(receptor.get_chain('alpha').amino_acid_sequence)
                fields["cdr3_a_nucseq"].append(receptor.get_chain("alpha").nucleotide_sequence)
            elif type(receptor).__name__ == "ReceptorSequence" and receptor.metadata.chain.value == "TRA":
                fields["v_a_gene"].append(TCRdistHelper.add_default_allele_to_v_gene(receptor.metadata.v_allele))
                fields["j_a_gene"].append(receptor.metadata.j_allele)
                fields["cdr3_a_aa"].append(receptor.amino_acid_sequence)
                fields["cdr3_a_nucseq"].append(receptor.nucleotide_sequence)

    @staticmethod
    def process_receptor_beta_chain(receptor, fields, chains):
        if "beta" in chains:
            if type(receptor).__name__ == "TCABReceptor":
                fields["v_b_gene"].append(TCRdistHelper.add_default_allele_to_v_gene(receptor.get_chain('beta').metadata.v_allele))
                fields["j_b_gene"].append(receptor.get_chain('beta').metadata.j_allele)
                fields["cdr3_b_aa"].append(receptor.get_chain('beta').amino_acid_sequence)
                fields["cdr3_b_nucseq"].append(receptor.get_chain("beta").nucleotide_sequence)
            elif type(receptor).__name__ == "ReceptorSequence" and receptor.metadata.chain.value == "TRB":
                fields["v_b_gene"].append(TCRdistHelper.add_default_allele_to_v_gene(receptor.metadata.v_allele))
                fields["j_b_gene"].append(receptor.metadata.j_allele)
                fields["cdr3_b_aa"].append(receptor.amino_acid_sequence)
                fields["cdr3_b_nucseq"].append(receptor.nucleotide_sequence)

    @staticmethod
    def prepare_tcr_dist_fields(chains):
        fields = {
            "subject": [],
            "epitope": [],
            "count": [],
            "clone_id": []
        }
        if "alpha" in chains:
            fields["v_a_gene"] = []
            fields["j_a_gene"] = []
            fields["cdr3_a_aa"] = []
            fields["cdr3_a_nucseq"] = []
        if "beta" in chains:
            fields["v_b_gene"] = []
            fields["j_b_gene"] = []
            fields["cdr3_b_aa"] = []
            fields["cdr3_b_nucseq"] = []
        return fields
