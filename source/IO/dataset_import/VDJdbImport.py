import pandas as pd
from typing import List


from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.util.ImportHelper import ImportHelper


class VDJdbImport(DataImport):
    """
    Imports data from VDJdb format into a ReceptorDataset or SequenceDataset depending on the value of "paired" parameter or
    to RepertoireDataset (a set of repertoires consisting of a list of receptor sequences).

    Arguments:

        metadata_file: path to the metadata file, used only when importing repertoires, for receptor datasets,
            metadata information is located together with the sequence data so there is no need for additional file

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_vdjdb_dataset:
            format: VDJdb
            params:
                # these parameters have to be always specified:
                metadata_file: path/to/metadata.csv # csv file with fields filename, subject_id and arbitrary others which can be used as labels in analysis
                path: path/to/location/of/repertoire/files/ # all repertoire files need to be in the same folder to be loaded (they will be discovered based on the metadata file)
                result_path: path/where/to/store/imported/repertoires/ # immuneML imports data to optimized representation to speed up analysis so this defines where to store these new representation files
                # the following parameter have these default values so these need to be specified only if a different behavior is required
                paired: True # whether to import_dataset paired data: if true returns ReceptorDataset, and if false returns SequenceDataset
                column_mapping:
                    V: v_genes
                    J: j_genes
                    CDR3: sequence_aas
                    complex.id: sequence_identifiers
                region_type: CDR3
                separator: "\\t"

    """
    COLUMNS = ["V", "J", "Gene", "CDR3", "complex.id"]
    CUSTOM_COLUMNS = {"Epitope": "epitope", "Epitope gene": "epitope_gene", "Epitope species": "epitope_species"}


    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        vdjdb_params = DatasetImportParams.build_object(**params)
        if vdjdb_params.is_repertoire:
            dataset = VDJdbImport.load_repertoire_dataset(vdjdb_params, dataset_name)
        else:
            dataset = VDJdbImport.load_sequence_dataset(vdjdb_params, dataset_name)
        return dataset

    @staticmethod
    def load_repertoire_dataset(params: DatasetImportParams, dataset_name: str) -> Dataset:
        return ImportHelper.import_repertoire_dataset(VDJdbImport.preprocess_repertoire, params, dataset_name)

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:
        return ImportHelper.load_repertoire_as_dataframe(metadata, params)

    @staticmethod
    def load_sequence_dataset(params: DatasetImportParams, dataset_name: str) -> Dataset:
        return ImportHelper.import_sequence_dataset(VDJdbImport.import_items, params, dataset_name) #, paired=params.paired)



    @staticmethod
    def import_items(path, params: DatasetImportParams):
        if params.paired:
            sequences = VDJdbImport.import_paired_sequences(path)
        else:
            sequences = VDJdbImport.import_all_sequences(path)

        return sequences

    @staticmethod
    def import_paired_sequences(path) -> List[Receptor]:
        columns = VDJdbImport.COLUMNS + list(VDJdbImport.CUSTOM_COLUMNS.keys())
        df = pd.read_csv(path, sep="\t", usecols=columns, dtype=str)
        identifiers = df["complex.id"].unique()
        receptors = []

        for identifier in identifiers:
            receptor = VDJdbImport.import_receptor(df, identifier)
            receptors.append(receptor)

        return receptors

    @staticmethod
    def import_receptor(df, identifier) -> TCABReceptor:
        alpha_row = df.loc[(df["complex.id"] == identifier) & (df["Gene"] == "TRA")].iloc[0]
        beta_row = df.loc[(df["complex.id"] == identifier) & (df["Gene"] == "TRB")].iloc[0]

        alpha = VDJdbImport.import_sequence(alpha_row)
        beta = VDJdbImport.import_sequence(beta_row)

        return TCABReceptor(alpha=alpha,
                            beta=beta,
                            identifier=identifier,
                            metadata={**beta.metadata.custom_params})

    @staticmethod
    def import_all_sequences(path) -> ReceptorSequenceList:
        columns = VDJdbImport.COLUMNS + list(VDJdbImport.CUSTOM_COLUMNS.keys())
        df = pd.read_csv(path, sep="\t", usecols=columns)
        sequences = df.apply(VDJdbImport.import_sequence, axis=1).values
        return sequences

    @staticmethod
    def import_sequence(row):
        metadata = SequenceMetadata(v_gene=str(row["V"]) if "V" in row else None,
                                    j_gene=str(row["J"]) if "J" in row else None,
                                    chain=str(row["Gene"])[-1] if "Gene" in row else None,
                                    region_type="CDR3",
                                    custom_params={VDJdbImport.CUSTOM_COLUMNS[key]: row[key]
                                                   for key in VDJdbImport.CUSTOM_COLUMNS})
        sequence = ReceptorSequence(amino_acid_sequence=row["CDR3"], metadata=metadata, identifier=str(row["complex.id"]))
        return sequence
