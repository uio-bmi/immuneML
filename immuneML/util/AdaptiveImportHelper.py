import numpy as np
import pandas as pd

from immuneML import Constants
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper


class AdaptiveImportHelper:

    @staticmethod
    def preprocess_dataframe(dataframe: pd.DataFrame, params: DatasetImportParams):

        if "sequence" in dataframe.columns:
            dataframe['junction'] = [y[(84 - 3 * len(x)): 78] if is_valid_sequence_str(x) else ''
                                     for x, y in zip(dataframe['junction_aa'], dataframe['sequence'])]
            dataframe['cdr3'] = dataframe['junction'].str[3:-3]
        dataframe['cdr3_aa'] = dataframe['junction_aa'].str[1:-1]

        if "frame_type" in dataframe.columns:
            dataframe['vj_in_frame'] = (dataframe.frame_type.str.upper() == 'IN').astype(str).str[:1]
            dataframe['stop_codon'] = (dataframe.frame_type.str.upper() == 'STOP').astype(str).str[:1]
            dataframe['productive'] = dataframe.junction_aa.notnull().astype(str).str[:1]
            dataframe.drop(columns=['frame_type'], inplace=True)

        if 'duplicate_count' in dataframe.columns:
            dataframe.loc[dataframe['duplicate_count'].isna(), 'duplicate_count'] = -1
            dataframe.duplicate_count = dataframe.duplicate_count.astype(int)

        dataframe = AdaptiveImportHelper.parse_adaptive_germline_to_imgt(dataframe, params.organism)
        dataframe = set_locus_column(dataframe)

        return dataframe

    @staticmethod
    def parse_adaptive_germline_to_imgt(dataframe, organism):
        gene_name_replacement = pd.read_csv(
            EnvironmentSettings.root_path / "immuneML/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        gene_name_replacement = gene_name_replacement[gene_name_replacement.Species == organism]
        gene_name_replacement = dict(zip(gene_name_replacement.Adaptive, gene_name_replacement.IMGT))

        # remove C and extra 0 from gene name but not from allele (e.g., TCRBV03-01*01 -> TRBV3-1*01) to follow IMGT
        # naming
        germline_value_replacement = {**{"TCRB": "TRB", "TCRA": "TRA"},
                                      **{f"-0{i}": f"-{str(i)}" for i in range(10)},
                                      **{f"J0": "J", "V0": "V"}}

        return AdaptiveImportHelper.parse_germline(dataframe, gene_name_replacement, germline_value_replacement)

    @staticmethod
    def parse_germline(dataframe: pd.DataFrame, gene_name_replacement: dict, germline_value_replacement: dict):

        for gene in ["v", "j"]:
            if f"{gene}_call" in dataframe.columns:

                dataframe = replace_nans_with_empty_str(dataframe, f"{gene}_call")
                dataframe[f"{gene}_call"].replace(gene_name_replacement, regex=True, inplace=True)
                dataframe[f"{gene}_call"].replace(germline_value_replacement, regex=True, inplace=True)

                if f"{gene}_allele" in dataframe.columns:
                    rows_to_add_allele = ~dataframe[f'{gene}_call'].str.contains("\\*") & dataframe[
                        f"{gene}_allele"].astype(
                        str).str.contains("[0-9]{2}")
                    dataframe.loc[rows_to_add_allele, f"{gene}_call"] = \
                        dataframe.loc[rows_to_add_allele, lambda df: [f"{gene}_call", f"{gene}_allele"]].agg('*'.join,
                                                                                                             axis=1)
            elif f"{gene}_gene" in dataframe.columns and f"{gene}_allele" in dataframe.columns:
                dataframe = parse_allele(dataframe, gene)
                dataframe = parse_gene_column(dataframe, gene, gene_name_replacement, germline_value_replacement)
                make_gene_call_from_gene_and_allele(dataframe, gene)
            elif f"{gene}_gene" in dataframe.columns:
                dataframe = parse_gene_column(dataframe, gene, gene_name_replacement, germline_value_replacement)
                dataframe.rename(columns={f'{gene}_gene': f'{gene}_call'}, inplace=True)

        dataframe.drop(columns=['v_gene', 'j_gene', 'v_allele', 'j_allele'], inplace=True, errors='ignore')
        return dataframe


def set_locus_column(df: pd.DataFrame):
    if 'locus' in df.columns:
        df.locus = [Chain.get_chain_value(item) for item in df.locus]
    elif 'v_call' in df.columns:
        df['locus'] = [Chain.get_chain_value(item[:3]) for item in df.v_call]
    elif 'j_call' in df.columns:
        df['locus'] = [Chain.get_chain_value(item[:3]) for item in df.j_call]
    return df


def make_gene_call_from_gene_and_allele(df: pd.DataFrame, gene: str):
    gene_ids = df[f"{gene}_gene"] != ''
    non_allele_ids = df[f"{gene}_allele"] == ''
    df[f"{gene}_call"] = df[[f"{gene}_gene", f"{gene}_allele"]].agg('*0'.join, axis=1)
    df.loc[~gene_ids, f'{gene}_call'] = ''
    gene_set_allele_not = np.logical_and(gene_ids, non_allele_ids)
    df.loc[gene_set_allele_not, f"{gene}_call"] = df.loc[gene_set_allele_not, f"{gene}_gene"]
    return df


def parse_allele(df: pd.DataFrame, gene: str):
    if f"{gene}_allele" in df.columns:
        df[f"{gene}_allele"] = df[f"{gene}_allele"].astype(str)
        allele_set = np.logical_and(df[f"{gene}_allele"] != 'nan', df[f"{gene}_allele"] != '-1.0')
        df.loc[allele_set, f"{gene}_allele"] = df.loc[allele_set, f"{gene}_allele"].str[:-2]
        df.loc[~allele_set, f"{gene}_allele"] = ''
    return df


def parse_gene_column(df: pd.DataFrame, gene, gene_name_replacement, germline_value_replacement):
    df[f"{gene}_gene"].replace(germline_value_replacement, regex=True, inplace=True)
    df[f"{gene}_gene"].replace(gene_name_replacement, regex=True, inplace=True)
    return df


def replace_nans_with_empty_str(df: pd.DataFrame, col: str):
    df[col].replace('nan', '', inplace=True)
    return df


def is_valid_sequence_str(x):
    return isinstance(x, str) and x not in ["unresolved", "no data", "na", "unknown", 'nan']
