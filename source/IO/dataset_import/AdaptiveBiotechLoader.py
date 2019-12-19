import numpy as np
import pandas as pd

from source.IO.dataset_import.GenericLoader import GenericLoader
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings


class AdaptiveBiotechLoader(GenericLoader):

    def _read_preprocess_file(self, filepath, params):

        df = pd.read_csv(filepath,
                         sep="\t",
                         iterator=False,
                         usecols=["rearrangement", "v_family", "v_gene", "v_allele", "j_family", "j_gene",
                                  "j_allele", "amino_acid", "templates", "frame_type"],
                         dtype={"v_family": str,
                                "v_gene": str,
                                "v_allele": str,
                                "j_family": str,
                                "j_gene": str,
                                "j_allele": str,
                                "amino_acid": str,
                                "rearrangement": str,
                                "templates": str,
                                "frame_type": str})

        df = df.rename(columns={'rearrangement': 'sequences', 'amino_acid': 'sequence_aas', "v_gene": "v_genes", "j_gene": "j_genes",
                                "frame_type": "frame_types", 'v_family': 'v_subgroup', 'j_family': 'j_subgroup'})

        df = df.replace(["unresolved", "no data", "na", "unknown", "null", "nan", np.nan], Constants.UNKNOWN)
        if params["remove_out_of_frame"]:
            df = df[(df["sequence_aas"] != Constants.UNKNOWN) & (~df["sequence_aas"].str.contains("\*"))]
        df['sequences'] = [y[(84 - 3 * len(x)): 78] for x, y in zip(df['sequence_aas'], df['sequences'])]
        df['sequence_aas'] = df["sequence_aas"].str[1:-1]

        df = AdaptiveBiotechLoader.parse_germline(df)

        return df

    @staticmethod
    def parse_germline(df):
        replace_imgt = pd.read_csv(
            EnvironmentSettings.root_path + "source/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        replace_imgt = dict(zip(replace_imgt.Adaptive, replace_imgt.IMGT))

        df[["v_genes", "j_genes"]] = df[["v_genes", "j_genes"]].replace(replace_imgt)

        replace_dict = {"TCRB": "TRB"}

        replace_dict = {**replace_dict,
                        **{("0" + str(i)): str(i) for i in range(10)}}

        df[["v_subgroup", "v_genes", "j_subgroup", "j_genes"]] = df[
            ["v_subgroup", "v_genes", "j_subgroup", "j_genes"]].replace(replace_dict, regex=True)

        df["v_allele"] = df['v_genes'].str.cat(df['v_allele'], sep=Constants.ALLELE_DELIMITER)
        df["j_allele"] = df['j_genes'].str.cat(df['j_allele'], sep=Constants.ALLELE_DELIMITER)

        return df
