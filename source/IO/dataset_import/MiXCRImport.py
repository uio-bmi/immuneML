import pandas as pd

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset import Dataset
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class MiXCRImport(DataImport):
    """
    Imports repertoire files into immuneML format from the repertoire tsv files that were generated as
    results of MiXCR preprocessing.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_mixcr_dataset:
            format: MiXCR
            params:
                # these parameters have to be always specified:
                metadata_file: path/to/metadata.csv # csv file with fields filename, subject_id and arbitrary others which can be used as labels in analysis
                path: path/to/location/of/repertoire/files/ # all repertoire files need to be in the same folder to be loaded (they will be discovered based on the metadata file)
                result_path: path/where/to/store/imported/repertoires/ # immuneML imports data to optimized representation to speed up analysis so this defines where to store these new representation files
                # the following parameter have these default values so these need to be specified only if a different behavior is required
                region_type: "IMGT_CDR3" # which part of the sequence to import by default
                batch_size: 4 # how many repertoires can be processed at once by default
                separator: "\\t"
                columns_to_load: [cloneCount, allVHitsWithScore, allJHitsWithScore, aaSeqCDR3, nSeqCDR3]
                column_mapping: # MiXCR column name -> immuneML repertoire field (where there is no 1-1 mapping, those are omitted here and handled in the code)
                    cloneCount: counts
                    allVHitsWithScore: v_genes
                    allJHitsWithScore: j_genes

    """

    SEQUENCE_NAME_MAP = {
        RegionType.IMGT_CDR3: {"AA": "aaSeqCDR3", "NT": "nSeqCDR3"},
        RegionType.IMGT_CDR1: {"AA": "aaSeqCDR1", "NT": "nSeqCDR1"},
        RegionType.IMGT_CDR2: {"AA": "aaSeqCDR2", "NT": "nSeqCDR2"},
        RegionType.IMGT_FR1:  {"AA": "aaSeqFR1", "NT": "nSeqFR1"},
        RegionType.IMGT_FR2:  {"AA": "aaSeqFR2", "NT": "nSeqFR2"},
        RegionType.IMGT_FR3:  {"AA": "aaSeqFR3", "NT": "nSeqFR3"},
        RegionType.IMGT_FR4:  {"AA": "aaSeqFR4", "NT": "nSeqFR4"}
    }


    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(MiXCRImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        """
        Function for loading the data from one MiXCR file, such that:
            - for the given region (CDR3/full sequence), both nucleotide and amino acid sequence are loaded
            - if the region is CDR3, it adapts the sequence to the definition of the CDR3 (IMGT junction vs IMGT CDR3)
            - the chain for each sequence is extracted from the v gene name
            - the genes are loaded from the top score for gene without allele info

        Arguments:

            metadata: the corresponding row from the metadata file with metadata such as subject_id, age, HLA or other info given there
            params: DatasetImportParams object defining what to import and how to do it

        Returns:
            data frame corresponding to Repertoire.FIELDS and custom lists which can be used to create a Repertoire object

        """
        df["sequence_aas"] = df[MiXCRImport.SEQUENCE_NAME_MAP[params.region_type]["AA"]]
        df["sequences"] = df[MiXCRImport.SEQUENCE_NAME_MAP[params.region_type]["NT"]]
        ImportHelper.junction_to_cdr3(df, params.region_type)

        df["counts"] = df["counts"].astype(float).astype(int)

        if "v_genes" in df.columns:
            df["chains"] = ImportHelper.load_chains_from_genes(df, "v_genes")
            df["v_genes"] = MiXCRImport._load_genes(df, "v_genes")
        if "j_genes" in df.columns:
            if "chains" not in df.columns:
                df["chains"] = ImportHelper.load_chains_from_genes(df, "j_genes")
            df["j_genes"] = MiXCRImport._load_genes(df, "j_genes")

        return df


    @staticmethod
    def _load_genes(df: pd.DataFrame, column_name):
        # note: MiXCR omits the '/' for 'TRA.../DV' genes
        tmp_df = df.apply(lambda row: row[column_name].split(",")[0].replace("DV", "/DV").replace("//", "/").split("*", 1)[0], axis=1)

        return tmp_df
