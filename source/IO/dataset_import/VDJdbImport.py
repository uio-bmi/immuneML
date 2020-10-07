import pandas as pd


from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
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
                    Gene: chains
                region_type: CDR3
                separator: "\\t" # todo look at this after refactoring

    """
    COLUMNS = ["V", "J", "Gene", "CDR3", "complex.id"]
    CUSTOM_COLUMNS = {"Epitope": "epitope", "Epitope gene": "epitope_gene", "Epitope species": "epitope_species"}


    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(VDJdbImport, params, dataset_name)


    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:
        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)
        df = VDJdbImport.preprocess_dataframe(df, params)
        return df


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        df["frame_types"] = SequenceFrameType.IN.name
        return df


    @staticmethod
    def import_receptors(df, params):
        df["receptor_identifiers"] = df["sequence_identifiers"]
        return ImportHelper.import_receptors(df, params)



