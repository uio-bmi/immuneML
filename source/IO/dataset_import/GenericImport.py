from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class GenericImport(DataImport):

    @staticmethod
    def import_dataset(params: DatasetImportParams) -> RepertoireDataset:
        dataset = ImportHelper.import_repertoire_dataset(GenericImport.preprocess_repertoire, params)
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
