import functools
import operator

import pandas as pd

from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.environment.Constants import Constants


class IRISSequenceImport:

    @staticmethod
    def import_sequences(path: str):
        df = pd.read_csv(path, sep=";")
        df = df.where((pd.notnull(df)), None)

        sequences = df.apply(IRISSequenceImport.process_iris_row, axis=1).values
        sequences = functools.reduce(operator.iconcat, sequences, [])

        return sequences

    @staticmethod
    def process_iris_row(row):
        sequences = []

        if row["Chain: TRA (1)"] is not None:
            sequences.extend(IRISSequenceImport.process_iris_chain(row, "A"))
        if row["Chain: TRB (1)"] is not None:
            sequences.extend(IRISSequenceImport.process_iris_chain(row, "B"))

        return sequences

    @staticmethod
    def process_iris_chain(row, chain):
        sequences = []

        v_genes = set([gene.split(Constants.ALLELE_DELIMITER)[0].replace("TR{}".format(chain), "").replace(chain, "") for gene in
                       row["TR{} - V gene (1)".format(chain)].split(" | ")])
        j_genes = set([gene.split(Constants.ALLELE_DELIMITER)[0].replace("TR{}".format(chain), "").replace(chain, "") for gene in
                       row["TR{} - J gene (1)".format(chain)].split(" | ")])

        for v_gene in v_genes:
            for j_gene in j_genes:
                metadata = SequenceMetadata(v_gene=v_gene, j_gene=j_gene, chain=chain)
                sequences.append(ReceptorSequence(amino_acid_sequence=row["Chain: TR{} (1)".format(chain)], metadata=metadata))

        return sequences
