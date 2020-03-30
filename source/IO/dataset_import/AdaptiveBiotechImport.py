import numpy as np

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class AdaptiveBiotechImport(DataImport):

    @staticmethod
    def import_dataset(params: DatasetImportParams) -> RepertoireDataset:
        dataset = ImportHelper.import_repertoire_dataset(AdaptiveBiotechImport.preprocess_repertoire, params)
        return dataset

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:

        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)

        frame_type_list = AdaptiveBiotechImport.prepare_frame_type_list(params)
        df = df[df["frame_types"].isin(frame_type_list)]

        if params.region_definition == RegionDefinition.IMGT:
            if "sequences" in params.columns_to_load:
                df['sequences'] = [y[(84 - 3 * len(x)): 78] for x, y in zip(df['sequence_aas'], df['sequences'])]
            df['sequence_aas'] = df["sequence_aas"].str[1:-1]
        elif "sequences" in params.columns_to_load:
            df['sequences'] = [y[(81 - 3 * len(x)): 81] for x, y in zip(df['sequence_aas'], df['sequences'])]

        df = ImportHelper.parse_adaptive_germline_to_imgt(df)

        df["chains"] = np.where(df["v_genes"].isnull(), df["j_genes"].str[:3], df["v_genes"].str[:3])
        df["region_types"] = RegionType.CDR3.name

        return df

    @staticmethod
    def prepare_frame_type_list(params: DatasetImportParams) -> list:
        frame_type_list = []
        if params.import_productive:
            frame_type_list.append("In")
        if params.import_out_of_frame:
            frame_type_list.append("Out")
        if params.import_with_stop_codon:
            frame_type_list.append("Stop")
        return frame_type_list
