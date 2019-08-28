import pandas as pd

from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata


class VDJdbSequenceImport:
    """
    Loads in the data to a list of sequences from VDJdb format
    """

    COLUMNS = ["V", "J", "Gene", "CDR3", "complex.id"]

    @staticmethod
    def import_sequences(path, paired: bool = False):
        if paired:
            sequences = VDJdbSequenceImport.import_paired_sequences(path)
        else:
            sequences = VDJdbSequenceImport.import_all_sequences(path, VDJdbSequenceImport.COLUMNS)

        return sequences

    @staticmethod
    def import_paired_sequences(path, columns) -> list:
        df = pd.read_csv(path, sep="\t", usecols=columns)
        identifiers = df["complex.id"].unique()
        receptors = []

        for identifier in identifiers:
            receptor = VDJdbSequenceImport.import_receptor(df, identifier)
            receptors.append(receptor)

        return receptors

    @staticmethod
    def import_receptor(df, identifier) -> TCABReceptor:
        alpha_row = df.loc[(df["complex.id"] == identifier) & (df["Gene"] == "TRA")].iloc[0]
        beta_row = df.loc[(df["complex.id"] == identifier) & (df["Gene"] == "TRB")].iloc[0]
        return TCABReceptor(alpha=VDJdbSequenceImport.import_sequence(alpha_row),
                            beta=VDJdbSequenceImport.import_sequence(beta_row),
                            identifier=identifier)

    @staticmethod
    def import_all_sequences(path, columns: list = None) -> list:
        df = pd.read_csv(path, sep="\t", usecols=columns)
        sequences = df.apply(VDJdbSequenceImport.import_sequence, axis=1).values
        return sequences

    @staticmethod
    def import_sequence(row):
        metadata = SequenceMetadata(v_gene=row["V"][3:],  # remove TRB/A from gene name
                                    j_gene=row["J"][3:],  # remove TRB/A from gene name
                                    chain=row["Gene"][-1],
                                    region_type="CDR3")
        sequence = ReceptorSequence(amino_acid_sequence=row["CDR3"], metadata=metadata)
        return sequence
