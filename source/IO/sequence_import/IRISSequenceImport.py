import functools
import operator

import pandas as pd

from source.data_model.receptor.ReceptorList import ReceptorList
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.environment.Constants import Constants


class IRISSequenceImport:

    @staticmethod
    def import_items(path: str, paired: bool = False):
        df = pd.read_csv(path, sep=";")
        df = df.where((pd.notnull(df)), None)

        sequences = df.apply(IRISSequenceImport.process_iris_row, paired=paired, axis=1).values
        sequences = functools.reduce(operator.iconcat, sequences, [])

        return sequences

    @staticmethod
    def process_iris_row(row, paired):
        if paired:
            sequences = ReceptorList()

            alpha_chain = IRISSequenceImport.process_iris_chain(row, "A")
            beta_chain = IRISSequenceImport.process_iris_chain(row, "B")

            # Only uses the first alpha/beta chain, dual chains are ignored
            sequences.extend([TCABReceptor(alpha=alpha_chain[0], beta=beta_chain[0], identifier=row["Clonotype ID"])])
        else:
            sequences = ReceptorSequenceList()
            if row["Chain: TRA (1)"] is not None:
                sequences.extend(IRISSequenceImport.process_iris_chain(row, "A"))
            if row["Chain: TRB (1)"] is not None:
                sequences.extend(IRISSequenceImport.process_iris_chain(row, "B"))

        return sequences

    @staticmethod
    def process_iris_chain(row, chain):
        sequences = ReceptorSequenceList()

        v_genes = set([gene.split(Constants.ALLELE_DELIMITER)[0].replace("TR{}".format(chain), "").replace(chain, "") for gene in
                       row["TR{} - V gene (1)".format(chain)].split(" | ")])
        j_genes = set([gene.split(Constants.ALLELE_DELIMITER)[0].replace("TR{}".format(chain), "").replace(chain, "") for gene in
                       row["TR{} - J gene (1)".format(chain)].split(" | ")])

        for v_gene in v_genes:
            for j_gene in j_genes:
                metadata = SequenceMetadata(v_gene=v_gene, j_gene=j_gene, chain=chain)
                sequences.append(ReceptorSequence(amino_acid_sequence=row["Chain: TR{} (1)".format(chain)], metadata=metadata))

        return sequences
