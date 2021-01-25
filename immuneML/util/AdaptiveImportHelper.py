import pandas as pd

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper


class AdaptiveImportHelper:

    @staticmethod
    def preprocess_dataframe(dataframe: pd.DataFrame, params: DatasetImportParams):
        dataframe.loc[:, "frame_types"] = dataframe.frame_types.str.upper()

        frame_type_list = ImportHelper.prepare_frame_type_list(params)
        dataframe = dataframe[dataframe["frame_types"].isin(frame_type_list)]
        dataframe.loc[:, "region_types"] = params.region_type.name

        if params.region_type == RegionType.IMGT_CDR3:
            if "sequences" in dataframe.columns:
                dataframe.loc[:, 'sequences'] = [y[(84 - 3 * len(x)): 78] if x is not None else None for x, y in zip(dataframe['sequence_aas'], dataframe['sequences'])]
            dataframe.loc[:, 'sequence_aas'] = dataframe["sequence_aas"].str[1:-1]
        elif "sequences" in dataframe.columns:
            dataframe.loc[:, 'sequences'] = [y[(81 - 3 * len(x)): 81] if x is not None else None for x, y in zip(dataframe['sequence_aas'], dataframe['sequences'])]

        dataframe = AdaptiveImportHelper.parse_adaptive_germline_to_imgt(dataframe, params.organism)
        dataframe = ImportHelper.standardize_none_values(dataframe)
        ImportHelper.drop_empty_sequences(dataframe, params.import_empty_aa_sequences, params.import_empty_nt_sequences)
        ImportHelper.drop_illegal_character_sequences(dataframe, params.import_illegal_characters)

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
    def parse_adaptive_germline_to_imgt(dataframe, organism):
        gene_name_replacement = pd.read_csv(
            EnvironmentSettings.root_path / "immuneML/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        gene_name_replacement = gene_name_replacement[gene_name_replacement.Species == organism]
        gene_name_replacement = dict(zip(gene_name_replacement.Adaptive, gene_name_replacement.IMGT))

        # remove C and extra 0 from gene name but not from allele (e.g., TCRBV03-01*01 -> TRBV3-1*01) to follow IMGT naming
        germline_value_replacement = {**{"TCRB": "TRB", "TCRA": "TRA"},
                                      **{f"-0{i}": f"-{str(i)}" for i in range(10)},
                                      **{f"J0": "J", "V0": "V"}}

        return AdaptiveImportHelper.parse_germline(dataframe, gene_name_replacement, germline_value_replacement)

    @staticmethod
    def parse_germline(dataframe: pd.DataFrame, gene_name_replacement: dict, germline_value_replacement: dict):

        for column in ["v_genes", "j_genes"]:
            dataframe.loc[:, column] = dataframe[column].replace(gene_name_replacement, regex=True)

        for column in ["v_subgroups", "v_genes", "j_subgroups", "j_genes"]:
            if column in dataframe.columns:
                dataframe.loc[:, column] = dataframe[column].replace(germline_value_replacement, regex=True)

        for col_gene, col_allele in [["v_genes", "v_alleles"], ["j_genes", "j_alleles"]]:
            if col_allele in dataframe.columns and col_gene in dataframe.columns:
                dataframe.loc[:, col_allele] = dataframe[col_gene].str.cat([Constants.ALLELE_DELIMITER + item.split(Constants.ALLELE_DELIMITER)[-1]
                                                                     if item is not None and Constants.ALLELE_DELIMITER in item else '' for item in dataframe[col_allele]])
            elif col_gene in dataframe.columns:
                dataframe.loc[:, col_allele] = dataframe[col_gene].copy()

        return dataframe
