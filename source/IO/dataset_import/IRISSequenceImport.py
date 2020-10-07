import functools
import itertools as it
import operator

import pandas as pd

from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.environment.Constants import Constants


class IRISSequenceImport:
    """
    Reads immune receptor data in IRIS format, and reads this into a list of Receptors (paired alpha-beta)
    or ReceptorSequences (unpaired).

    By default, when dual chain information is present and when multiple possible
    genes are specified for a ReceptorSequence, all possible combinations are read in.

    Illegal rows containing missing information are skipped without warning.

    Note: currenly only works with TCR alpha-beta data
    # todo generalize to BCR data, and TCR gamma-delta


    Arguments:
        path (str): Path to the IRIS file.
        paired (bool): Determines whether the data should be read as paired or unpaired format. When paired is True,
            the class returns a ReceptorList. When paired is False, it returns a ReceptorSequenceList.
        all_dual_chains (bool): Determines whether all dual chain information should be read in. When all_dual_chains
            is True, both chain (1) and (2) are read in. In combination with paired is True, this means all possible
            combinations of alpha and beta chains are added to the ReceptorList, and when paired is False all possible
            single chains are added to the ReceptorSequenceList.
            When all_dual_chains is False, only chain (1) is read in and (2) is ignored, creating only one entry in the
            ReceptorList or ReceptorSequenceList per line in the IRIS file.
            Whether a ReceptorSequence was chain (1) or (2) is stored in the custom_params list named dual_chain_id,
            and in the case of paired data the information is additionally present in the metadata list clonotype_id.
        all_genes (bool): Determines whether all genes should be read in, when multiple gene options are present
            (separated by the symbol |). Similarly to all_dual_chains, when all_genes is True, all versions of a given
            ReceptorSequence are considered. When all_genes is False and multiple genes are present, a random gene is
            selected from the set.


    Specification:

        datasets:
            my_dataset:
                path: /path/to/file.txt
                format: IRIS
                params:
                    paired: True
                    all_dual_chains: True
                    all_genes: True

    """

    @staticmethod
    def import_items(path: str, paired: bool = False, all_dual_chains: bool = True, all_genes: bool = False):
        df = pd.read_csv(path, sep="\t")
        df = df.where((pd.notnull(df)), None)

        sequences = df.apply(IRISSequenceImport.process_iris_row,
                             paired=paired,
                             all_dual_chains=all_dual_chains,
                             all_genes=all_genes, axis=1).values
        sequences = functools.reduce(operator.iconcat, sequences, [])

        return sequences

    @staticmethod
    def process_iris_row(row, paired: bool = False, all_dual_chains: bool = True, all_genes: bool = False):
        if paired:
            sequences = []
            if row["Chain: TRA (1)"] is not None and row["Chain: TRB (1)"] is not None:
                alpha_seqs = IRISSequenceImport.process_iris_chain(row, "A", 1, all_genes)
                beta_seqs = IRISSequenceImport.process_iris_chain(row, "B", 1, all_genes)

                if all_dual_chains:
                    if row["Chain: TRA (2)"] is not None:
                        alpha_seqs.extend(IRISSequenceImport.process_iris_chain(row, "A", 2, all_genes))
                    if row["Chain: TRB (2)"] is not None:
                        beta_seqs.extend(IRISSequenceImport.process_iris_chain(row, "B", 2, all_genes))

                for alpha_i, alpha_seq in enumerate(alpha_seqs):
                    for beta_i, beta_seq in enumerate(beta_seqs):
                        clonotype_id = row["Clonotype ID"]
                        identifier = f"{clonotype_id}-A{alpha_i}-B{beta_i}"
                        sequences.extend([TCABReceptor(alpha=alpha_seq,
                                                       beta=beta_seq,
                                                       identifier=identifier,
                                                       metadata={"clonotype_id": clonotype_id})])
        else:
            sequences = ReceptorSequenceList()
            # process all dual chains if specified, otherwise just chain 1
            to_process = list(it.product(["A", "B"], [1, 2] if all_dual_chains else [1]))

            for chain, dual_chain_id in to_process:
                if row[f"Chain: TR{chain} ({dual_chain_id})"] is not None:
                    sequences.extend(IRISSequenceImport.process_iris_chain(row, chain, dual_chain_id, all_genes))

        return sequences

    @staticmethod
    def process_iris_chain(row, chain, dual_chain_id, all_genes):
        sequences = ReceptorSequenceList()

        v_genes = set([gene.split(Constants.ALLELE_DELIMITER)[0].replace("TR{}".format(chain), "").replace(chain, "") for gene in
                       row["TR{} - V gene (1)".format(chain)].split(" | ")])
        j_genes = set([gene.split(Constants.ALLELE_DELIMITER)[0].replace("TR{}".format(chain), "").replace(chain, "") for gene in
                       row["TR{} - J gene (1)".format(chain)].split(" | ")])

        if all_genes:
            for v_gene in v_genes:
                    for j_gene in j_genes:
                        metadata = SequenceMetadata(v_gene=v_gene, j_gene=j_gene, chain=chain, custom_params={"dual_chain_id": dual_chain_id})
                        sequences.append(ReceptorSequence(amino_acid_sequence=row[f"Chain: TR{chain} ({dual_chain_id})"], metadata=metadata))
        else:
            # select a random v and j gene
            metadata = SequenceMetadata(v_gene=v_genes.pop(), j_gene=j_genes.pop(), chain=chain,
                                        custom_params={"dual_chain_id": dual_chain_id})
            sequences.append(ReceptorSequence(amino_acid_sequence=row[f"Chain: TR{chain} ({dual_chain_id})"], metadata=metadata))

        return sequences
