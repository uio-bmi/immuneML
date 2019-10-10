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

        df = df.rename(columns={'rearrangement': 'nucleotide', 'v_family': 'v_subgroup', 'j_family': 'j_subgroup'})

        df = df.replace(["unresolved", "no data", "na", "unknown", "null", "nan", np.nan], Constants.UNKNOWN)
        df['nucleotide'] = [y[(84 - 3 * len(x)): 78] for x, y in zip(df['amino_acid'], df['nucleotide'])]
        df['amino_acid'] = df["amino_acid"].str[1:-1]

        df = AdaptiveBiotechLoader.parse_germline(df)

        return df

    @staticmethod
    def parse_germline(df):
        replace_imgt = pd.read_csv(
            EnvironmentSettings.root_path + "source/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        replace_imgt = dict(zip(replace_imgt.Adaptive, replace_imgt.IMGT))

        df[["v_gene", "j_gene"]] = df[["v_gene", "j_gene"]].replace(replace_imgt)

        replace_dict = {"TCRB": "TRB"}

        replace_dict = {**replace_dict,
                        **{("0" + str(i)): str(i) for i in range(10)}}

        df[["v_subgroup", "v_gene", "j_subgroup", "j_gene"]] = df[
            ["v_subgroup", "v_gene", "j_subgroup", "j_gene"]].replace(replace_dict, regex=True)

        df["v_allele"] = df['v_gene'].str.cat(df['v_allele'], sep=Constants.ALLELE_DELIMITER)
        df["j_allele"] = df['j_gene'].str.cat(df['j_allele'], sep=Constants.ALLELE_DELIMITER)

        return df
