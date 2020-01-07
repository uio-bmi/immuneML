import pandas as pd

from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata


class VDJdbSequenceImport:
    """
    Loads in the data to a list of sequences from VDJdb format
    """

    COLUMNS = ["V", "J", "Gene", "CDR3", "complex.id"]
    CUSTOM_COLUMNS = {"Epitope": "epitope", "Epitope gene": "epitope_gene", "Epitope species": "epitope_species"}

    @staticmethod
    def import_items(path, paired: bool = False):
        if paired:
            sequences = VDJdbSequenceImport.import_paired_sequences(path)
        else:
            sequences = VDJdbSequenceImport.import_all_sequences(path)

        return sequences

    @staticmethod
    def import_paired_sequences(path) -> list:
        columns = VDJdbSequenceImport.COLUMNS + list(VDJdbSequenceImport.CUSTOM_COLUMNS.keys())
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

        alpha = VDJdbSequenceImport.import_sequence(alpha_row)
        beta = VDJdbSequenceImport.import_sequence(beta_row)

        return TCABReceptor(alpha=alpha,
                            beta=beta,
                            identifier=identifier,
                            metadata=beta.metadata.custom_params)

    @staticmethod
    def import_all_sequences(path) -> list:
        columns = VDJdbSequenceImport.COLUMNS + list(VDJdbSequenceImport.CUSTOM_COLUMNS.keys())
        df = pd.read_csv(path, sep="\t", usecols=columns)
        sequences = df.apply(VDJdbSequenceImport.import_sequence, axis=1).values
        return sequences

    @staticmethod
    def import_sequence(row):
        metadata = SequenceMetadata(v_gene=row["V"][3:] if "V" in row else None,  # remove TRB/A from gene name
                                    j_gene=row["J"][3:] if "J" in row else None,  # remove TRB/A from gene name
                                    chain=row["Gene"][-1] if "Gene" in row else None,
                                    region_type="CDR3",
                                    custom_params={VDJdbSequenceImport.CUSTOM_COLUMNS[key]: row[key]
                                                   for key in VDJdbSequenceImport.CUSTOM_COLUMNS})
        sequence = ReceptorSequence(amino_acid_sequence=row["CDR3"], metadata=metadata, identifier=str(row["complex.id"]))
        return sequence
