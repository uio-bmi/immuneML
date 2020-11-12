import pandas as pd

from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.receptor.RegionType import RegionType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.ImportHelper import ImportHelper


class AdaptiveImportHelper:

    @staticmethod
    def preprocess_dataframe(dataframe: pd.DataFrame, params: DatasetImportParams):
        dataframe["frame_types"] = dataframe.frame_types.str.upper()

        frame_type_list = ImportHelper.prepare_frame_type_list(params)
        dataframe = dataframe[dataframe["frame_types"].isin(frame_type_list)]
        dataframe["region_types"] = params.region_type.name

        if params.region_type == RegionType.IMGT_CDR3:
            if "sequences" in dataframe.columns:
                dataframe['sequences'] = [y[(84 - 3 * len(x)): 78] if x is not None else None for x, y in zip(dataframe['sequence_aas'], dataframe['sequences'])]
            dataframe['sequence_aas'] = dataframe["sequence_aas"].str[1:-1]
        elif "sequences" in dataframe.columns:
            dataframe['sequences'] = [y[(81 - 3 * len(x)): 81] if x is not None else None for x, y in zip(dataframe['sequence_aas'], dataframe['sequences'])]

        dataframe = AdaptiveImportHelper.parse_adaptive_germline_to_imgt(dataframe)
        dataframe = ImportHelper.standardize_none_values(dataframe)
        ImportHelper.drop_empty_sequences(dataframe, params.import_empty_aa_sequences, params.import_empty_nt_sequences)

        if "chains" in dataframe.columns:
            dataframe.loc[:, "chains"] = ImportHelper.load_chains(dataframe)
        else:
            # loading from v_subgroups is preferred as sometimes v_genes is None when v_subgroups is defined
            if "v_subgroups" in dataframe.columns:
                dataframe.loc[:, "chains"] = ImportHelper.load_chains_from_column(dataframe, "v_subgroups")
            else:
                dataframe.loc[:, "chains"] = ImportHelper.load_chains_from_genes(dataframe)

        return dataframe

    @staticmethod
    def parse_adaptive_germline_to_imgt(dataframe):
        gene_name_replacement = pd.read_csv(
            EnvironmentSettings.root_path + "source/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        gene_name_replacement = dict(zip(gene_name_replacement.Adaptive, gene_name_replacement.IMGT))

        germline_value_replacement = {**{"TCRB": "TRB", "TCRA": "TRA"}, **{("0" + str(i)): str(i) for i in range(10)}}

        return AdaptiveImportHelper.parse_germline(dataframe, gene_name_replacement, germline_value_replacement)

    @staticmethod
    def parse_germline(dataframe: pd.DataFrame, gene_name_replacement: dict, germline_value_replacement: dict):

        if all(item in dataframe.columns for item in ["v_genes", "j_genes"]):
            dataframe[["v_genes", "j_genes"]] = dataframe[["v_genes", "j_genes"]].replace(gene_name_replacement)

        if all(item in dataframe.columns for item in ["v_subgroups", "v_genes", "j_subgroups", "j_genes"]):
            dataframe[["v_subgroups", "v_genes", "j_subgroups", "j_genes"]] = dataframe[
                ["v_subgroups", "v_genes", "j_subgroups", "j_genes"]].replace(germline_value_replacement, regex=True)

        if all(item in dataframe.columns for item in ["v_genes", "j_genes", "v_alleles", "j_alleles"]):
            dataframe["v_alleles"] = dataframe['v_genes'].str.cat(dataframe['v_alleles'], sep=Constants.ALLELE_DELIMITER)
            dataframe["j_alleles"] = dataframe['j_genes'].str.cat(dataframe['j_alleles'], sep=Constants.ALLELE_DELIMITER)

        return dataframe