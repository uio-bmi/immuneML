import pandas as pd

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper


class AdaptiveImportHelper:

    @staticmethod
    def preprocess_dataframe(dataframe: pd.DataFrame, params: DatasetImportParams):
        if "frame_type" in dataframe.columns:
            dataframe.loc[:, "frame_type"] = dataframe.frame_type.str.upper()

            frame_type_list = ImportHelper.prepare_frame_type_list(params)
            dataframe = dataframe[dataframe["frame_type"].isin(frame_type_list)]

        dataframe.loc[:, "region_type"] = params.region_type.name

        if params.region_type == RegionType.IMGT_CDR3:
            if "sequence" in dataframe.columns:
                dataframe.loc[:, 'sequence'] = [y[(84 - 3 * len(x)): 78] if x is not None else None for x, y in
                                                 zip(dataframe['sequence_aa'], dataframe['sequence'])]
            dataframe.loc[:, 'sequence_aa'] = dataframe["sequence_aa"].str[1:-1]
        elif "sequences" in dataframe.columns:
            dataframe.loc[:, 'sequence'] = [y[(81 - 3 * len(x)): 81] if x is not None else None for x, y in
                                             zip(dataframe['sequence_aa'], dataframe['sequence'])]

        dataframe = AdaptiveImportHelper.parse_adaptive_germline_to_imgt(dataframe, params.organism)
        ImportHelper.load_chains(dataframe)
        ImportHelper.drop_empty_sequences(dataframe, params.import_empty_aa_sequences, params.import_empty_nt_sequences)
        ImportHelper.drop_illegal_character_sequences(dataframe, params.import_illegal_characters, params.import_with_stop_codon)

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
        for gene in ["v", "j"]:
            for part in ['gene', 'allele', 'call', 'family']:
                if f"{gene}_{part}" in dataframe.columns:
                    dataframe.loc[:, f"{gene}_{part}"] = dataframe[f"{gene}_{part}"].replace(gene_name_replacement, regex=True)
                    if part != 'family':
                        dataframe.loc[:, f"{gene}_{part}"] = dataframe[f"{gene}_{part}"].replace(germline_value_replacement, regex=True)

            if f"{gene}_call" in dataframe.columns and f"{gene}_allele" in dataframe.columns:
                rows_to_add_allele = ~dataframe[f'{gene}_call'].str.contains("\*") & dataframe[f"{gene}_allele"].astype(str).str.contains("[0-9]{2}")
                dataframe.loc[rows_to_add_allele, f"{gene}_call"] = \
                    dataframe.loc[rows_to_add_allele, lambda df: [f"{gene}_call", f"{gene}_allele"]].agg('*'.join, axis=1)

        return dataframe
