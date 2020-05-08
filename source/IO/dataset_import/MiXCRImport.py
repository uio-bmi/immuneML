import pandas as pd

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.util.ImportHelper import ImportHelper


class MiXCRImport(DataImport):

    SEQUENCE_NAME_MAP = {
        RegionType.CDR3: {"AA": "aaSeqCDR3", "NT": "nSeqCDR3"},
        RegionType.CDR1: {"AA": "aaSeqCDR1", "NT": "nSeqCDR1"},
        RegionType.CDR2: {"AA": "aaSeqCDR2", "NT": "nSeqCDR2"},
        RegionType.FR1:  {"AA": "aaSeqFR1",  "NT": "nSeqFR1"},
        RegionType.FR2:  {"AA": "aaSeqFR2",  "NT": "nSeqFR2"},
        RegionType.FR3:  {"AA": "aaSeqFR3",  "NT": "nSeqFR3"},
        RegionType.FR4:  {"AA": "aaSeqFR4",  "NT": "nSeqFR4"}
    }

    @staticmethod
    def import_dataset(params: dict) -> RepertoireDataset:
        mixcr_params = DatasetImportParams.build_object(**params)
        dataset = ImportHelper.import_or_load_imported(params, mixcr_params, MiXCRImport.preprocess_repertoire)
        return dataset

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> pd.DataFrame:
        """
        Function for loading the data from one MiXCR file, such that:
            - for the given region (CDR3/full sequence), both nucleotide and amino acid sequence are loaded
            - if the region is CDR3, it adapts the sequence to the definition of the CDR3 (IMGT junction vs IMGT CDR3)
            - the chain for each sequence is extracted from the v gene name
            - the genes are loaded from the top score for gene without allele info
        Arguments:
            metadata: the corresponding row from the metadata file with metadata such as donor info, age, HLA or other info given there
            params: DatasetImportParams object defining what to import and how to do it
        Returns:
            data frame corresponding to Repertoire.FIELDS and custom lists which can be used to create a Repertoire object
        """

        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)

        sequences_aas = df[MiXCRImport.SEQUENCE_NAME_MAP[params.region_type]["AA"]]
        sequences = df[MiXCRImport.SEQUENCE_NAME_MAP[params.region_type]["NT"]]
        if params.region_definition == RegionDefinition.IMGT and params.region_type == RegionType.CDR3:
            sequences_aas = sequences_aas.str[1:-1]
            sequences = sequences.str[3:-3]

        df["region_types"] = params.region_type.name
        df["sequence_aas"] = sequences_aas
        df["sequences"] = sequences
        if "v_genes" in df.columns:
            df["chains"] = MiXCRImport._load_chains(df, "v_genes").tolist()
            df["v_genes"] = MiXCRImport._load_genes(df, "v_genes")
        if "j_genes" in df.columns:
            if "chains" not in df.columns:
                df["chains"] = MiXCRImport._load_chains(df, "j_genes").tolist()
            df["j_genes"] = MiXCRImport._load_genes(df, "j_genes")

        return df

    @staticmethod
    def _load_chains(df: pd.DataFrame, column_name):
        tmp_df = df.apply(lambda row: Chain[[x for x in [chain.value for chain in Chain] if x in row[column_name]][0]]
                          if len([x for x in [chain.value for chain in Chain] if x in row[column_name]]) > 0 else None, axis=1)
        return tmp_df

    @staticmethod
    def _load_genes(df: pd.DataFrame, column_name):
        # note: MiXCR omits the '/' for 'TRA.../DV' genes
        tmp_df = df.apply(lambda row: row[column_name].split(",")[0].replace("TRB", "").replace("TRA", "").replace("DV", "/DV").replace("//", "/").split("*", 1)[0], axis=1)

        return tmp_df
