from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class GenericImport(DataImport):
    """
    Imports repertoire files into immuneML format. It has the same parameters but no predefined default values,
    so everything needs to be specified manually.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_generic_dataset:
            format: Generic
            params:
                metadata_file: path/to/metadata.csv # csv file with fields filename, donor and arbitrary others which can be used as labels in analysis
                path: path/to/location/of/repertoire/files/ # all repertoire files need to be in the same folder to be loaded (they will be discovered based on the metadata file)
                result_path: path/where/to/store/imported/repertoires/ # immuneML imports data to optimized representation to speed up analysis so this defines where to store these new representation files
                region_type: "CDR3" # which part of the sequence to import by default
                batch_size: 4 # how many repertoires can be processed at once by default
                region_definition: "IMGT" # which CDR3 definition to use - IMGT option means removing first and last amino acid as formats like MiXCR or Adaptive's use IMGT junction as CDR3
                separator: "\\t"
                columns_to_load: [name_of_column_1_from_original_file, name_of_column_2_from_original_file]
                column_mapping: # column name -> immuneML repertoire field (where there is no 1-1 mapping, those are omitted here and handled in the code)
                    name_of_column_1_from_original_file: sequences
                    name_of_column_2_from_original_file: sequence_aas

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> RepertoireDataset:
        generic_params = DatasetImportParams.build_object(**params)
        dataset = ImportHelper.import_repertoire_dataset(GenericImport.preprocess_repertoire, generic_params, dataset_name)
        return dataset

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:

        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)

        if params.region_type == RegionType.CDR3 and params.region_definition == RegionDefinition.IMGT:
            if "sequence_aas" in df.columns:
                df['sequence_aas'] = df["sequence_aas"].str[1:-1]
            if "sequences" in df.columns:
                df["sequences"] = df["sequences"].str[1:-1]

        return df
