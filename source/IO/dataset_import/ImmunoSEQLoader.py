import numpy as np
import pandas as pd

from source.IO.dataset_import.AdaptiveBiotechLoader import AdaptiveBiotechLoader
from source.IO.dataset_import.GenericLoader import GenericLoader
from source.environment.Constants import Constants


class ImmunoSEQLoader(GenericLoader):

    def _read_preprocess_file(self, filepath, params):
        df = pd.read_csv(filepath,
                         sep="\t",
                         iterator=False,
                         usecols=['nucleotide', 'aminoAcid', 'count (templates/reads)',
                                  'vFamilyName', 'vGeneName', 'vGeneAllele',
                                  'jFamilyName', 'jGeneName', 'jGeneAllele',
                                  'sequenceStatus'],
                         dtype={"nucleotide": str,
                                "aminoAcid": str,
                                "count (templates/reads)": int,
                                "vFamilyName": str,
                                "vGeneName": str,
                                "vGeneAllele": str,
                                "jFamilyName": str,
                                "jGeneName": str,
                                "jGeneAllele": str,
                                "sequenceStatus": str})

        df = df.rename(columns={'aminoAcid': 'sequence_aas',
                                'nucleotide': 'sequences',
                                "sequenceStatus": "frame_type",
                                "vFamilyName": "v_subgroup",
                                "vGeneName": "v_genes",
                                "vGeneAllele": "v_allele",
                                "jFamilyName": "j_subgroup",
                                "jGeneName": "j_genes",
                                "jGeneAllele": "j_allele",
                                'count (templates/reads)': 'templates'})

        df = df.replace(["unresolved", "no data", "na", "unknown", "null", "nan", np.nan], Constants.UNKNOWN)

        df['sequences'] = [y[(84 - 3 * len(x)): 78] for x, y in zip(df['sequence_aas'], df['sequences'])]
        df['sequence_aas'] = df["sequence_aas"].str[1:-1]

        df = AdaptiveBiotechLoader.parse_germline(df)

        return df
