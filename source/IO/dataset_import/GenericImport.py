import pandas as pd
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset import Dataset
from source.util.ImportHelper import ImportHelper


class GenericImport(DataImport):
    """
    Imports repertoire files into immuneML format. It has the same parameters but no predefined default values,
    so everything needs to be specified manually.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_generic_dataset:
            format: Generic
            params:
                metadata_file: path/to/metadata.csv # csv file with fields filename, subject_id and arbitrary others which can be used as labels in analysis
                path: path/to/location/of/repertoire/files/ # all repertoire files need to be in the same folder to be loaded (they will be discovered based on the metadata file)
                result_path: path/where/to/store/imported/repertoires/ # immuneML imports data to optimized representation to speed up analysis so this defines where to store these new representation files
                region_type: "IMGT_CDR3" # which part of the sequence to import by default
                batch_size: 4 # how many repertoires can be processed at once by default
                separator: "\\t"
                columns_to_load: [name_of_column_1_from_original_file, name_of_column_2_from_original_file]
                column_mapping: # column name -> immuneML repertoire field (where there is no 1-1 mapping, those are omitted here and handled in the code)
                    name_of_column_1_from_original_file: sequences
                    name_of_column_2_from_original_file: sequence_aas

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(GenericImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        ImportHelper.junction_to_cdr3(df, params.region_type)
        return df

    @staticmethod
    def import_receptors(df, params):
        df["receptor_identifiers"] = df["sequence_identifiers"]
        return ImportHelper.import_receptors(df, params)

