import pandas as pd
from tcrdist.repertoire import TCRrep

from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.ReceptorDataset import ReceptorDataset


class TCRdistHelper:

    @staticmethod
    def compute_tcr_dist(dataset: ReceptorDataset, labels: list, cores: int) -> TCRrep:
        return CacheHandler.memo_by_params((('dataset_identifier', dataset.identifier), ("type", "TCRrep")),
                                           lambda: TCRdistHelper._compute_tcr_dist(dataset, labels, cores))

    @staticmethod
    def _compute_tcr_dist(dataset: ReceptorDataset, labels: list, cores: int):
        df = TCRdistHelper.prepare_tcr_dist_dataframe(dataset, labels)
        tcr_rep = TCRrep(cell_df=df, chains=['alpha', 'beta'], organism=dataset.params["organism"], cpus=cores, infer_index_cols=False,
                         deduplicate=False, index_cols=['clone_id'])
        return tcr_rep

    @staticmethod
    def prepare_tcr_dist_dataframe(dataset: ReceptorDataset, labels: list) -> pd.DataFrame:
        if len(labels) > 1:
            raise NotImplementedError(f"TCRdist: multiple labels specified ({str(labels)[1:-1]}), but only single label binary class "
                                      f"is currently supported in immuneML.")
        label = labels[0]

        subject, epitope, count, v_a_gene, j_a_gene, cdr3_a_aa, v_b_gene, j_b_gene, cdr3_b_aa, clone_id, cdr3_b_nucseq, cdr3_a_nucseq = \
            [], [], [], [], [], [], [], [], [], [], [], []

        for receptor in dataset.get_data():
            subject.append(receptor.metadata["subject"] if "subject" in receptor.metadata else "sub" + receptor.identifier)
            epitope.append(receptor.metadata[label])
            count.append(receptor.get_chain("alpha").metadata.count
                         if receptor.get_chain("alpha").metadata.count == receptor.get_chain("beta").metadata.count
                            and receptor.get_chain("beta").metadata.count is not None else 1)
            v_a_gene.append(receptor.get_chain('alpha').metadata.v_gene)
            j_a_gene.append(receptor.get_chain('alpha').metadata.j_gene)
            cdr3_a_aa.append(receptor.get_chain('alpha').amino_acid_sequence)
            cdr3_a_nucseq.append(receptor.get_chain("alpha").nucleotide_sequence)
            v_b_gene.append(receptor.get_chain('beta').metadata.v_gene)
            j_b_gene.append(receptor.get_chain('beta').metadata.j_gene)
            cdr3_b_aa.append(receptor.get_chain('beta').amino_acid_sequence)
            cdr3_b_nucseq.append(receptor.get_chain("beta").nucleotide_sequence)
            clone_id.append(receptor.identifier)

        if all(item is not None for item in cdr3_a_nucseq) and all(item is not None for item in cdr3_b_nucseq):
            return pd.DataFrame({"subject": subject, "epitope": epitope, "count": count, "v_a_gene": v_a_gene, "j_a_gene": j_a_gene,
                                 "cdr3_a_aa": cdr3_a_aa, "v_b_gene": v_b_gene, "j_b_gene": j_b_gene, "cdr3_b_aa": cdr3_b_aa, "clone_id": clone_id,
                                 "cdr3_b_nucseq": cdr3_b_nucseq, "cdr3_a_nucseq": cdr3_a_nucseq})
        else:
            return pd.DataFrame({"subject": subject, "epitope": epitope, "count": count, "v_a_gene": v_a_gene, "j_a_gene": j_a_gene,
                                 "cdr3_a_aa": cdr3_a_aa, "v_b_gene": v_b_gene, "j_b_gene": j_b_gene, "cdr3_b_aa": cdr3_b_aa, "clone_id": clone_id})
