import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset import Dataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from scripts.specification_util import update_docs_per_mapping


class MiXCRImport(DataImport):
    """
    Imports data in MiXCR format into a Repertoire-, or SequenceDataset.
    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
    antigen specificity.


    **Specification arguments:**

    - path (str): For RepertoireDatasets, this is the path to a directory with MiXCR files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

    - is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset. By default, is_repertoire is set to True.

    - metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the MiXCR files included under the column 'filename' are imported into the RepertoireDataset. For setting SequenceDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence, such as '_', are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as MiXCR uses IMGT junction as CDR3. Alternatively to importing the CDR3 sequence, other region types can be specified here as well. Valid values for region_type are defined in MiXCRImport.SEQUENCE_NAME_MAP.

    - column_mapping (dict): A mapping from MiXCR column names to immuneML's internal data representation. The columns that specify the sequences to import are handled by the region_type parameter. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the MiXCR file, or using alternative column names). Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS. For MiXCR, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

                cloneCount: duplicate_count
                allVHitsWithScore: v_call
                allJHitsWithScore: j_call

    - column_mapping_synonyms (dict): This is a column mapping that can be used if a column could have alternative names. The formatting is the same as column_mapping. If some columns specified in column_mapping are not found in the file, the columns specified in column_mapping_synonyms are instead attempted to be loaded. For MiXCR format, there is no default column_mapping_synonyms.

    - columns_to_load (list): Specifies which subset of columns must be loaded from the MiXCR file. By default, this is: [cloneCount, allVHitsWithScore, allJHitsWithScore, aaSeqCDR3, nSeqCDR3]

    - metadata_column_mapping (dict): Specifies metadata for Sequence- and ReceptorDatasets. This should specify a mapping similar to column_mapping where keys are MiXCR column names and values are the names that are internally used in immuneML as metadata fields. These metadata fields can be used as prediction labels for Sequence- and ReceptorDatasets. This parameter can also be used to specify sequence-level metadata columns for RepertoireDatasets, which can be used by reports. To set prediction label metadata for RepertoireDatasets, see metadata_file instead. For MiXCR format, there is no default metadata_column_mapping.

    - separator (str): Column separator, for MiXCR this is by default "\\t".


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            datasets:
                my_mixcr_dataset:
                    format: MiXCR
                    params:
                        path: path/to/files/
                        is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                        metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                        metadata_column_mapping: # metadata column mapping MiXCR: immuneML for SequenceDataset
                            mixcrColumnName1: metadata_label1
                            mixcrColumnName2: metadata_label2
                        region_type: IMGT_CDR3 # what part of the sequence to import
                        import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                        import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                        import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                        # Optional fields with MiXCR-specific defaults, only change when different behavior is required:
                        separator: "\\t" # column separator
                        columns_to_load: # subset of columns to load, sequence columns are handled by region_type parameter
                        - cloneCount
                        - allVHitsWithScore
                        - allJHitsWithScore
                        - aaSeqCDR3
                        - nSeqCDR3
                        column_mapping: # column mapping MiXCR: immuneML
                            cloneCount: duplicate_count
                            allVHitsWithScore: v_call
                            allJHitsWithScore: j_call

    """

    SEQUENCE_NAME_MAP = {
        RegionType.IMGT_CDR3: {"AA": "aaSeqCDR3", "NT": "nSeqCDR3"},
        RegionType.IMGT_CDR1: {"AA": "aaSeqCDR1", "NT": "nSeqCDR1"},
        RegionType.IMGT_CDR2: {"AA": "aaSeqCDR2", "NT": "nSeqCDR2"},
        RegionType.IMGT_FR1: {"AA": "aaSeqFR1", "NT": "nSeqFR1"},
        RegionType.IMGT_FR2: {"AA": "aaSeqFR2", "NT": "nSeqFR2"},
        RegionType.IMGT_FR3: {"AA": "aaSeqFR3", "NT": "nSeqFR3"},
        RegionType.IMGT_FR4: {"AA": "aaSeqFR4", "NT": "nSeqFR4"}
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

            df: the dataframe as imported from the csv file

            params: DatasetImportParams object defining what to import and how to do it

        Returns:

            dataframe corresponding to Repertoire.FIELDS and custom lists which can be used to create a Repertoire object

        """
        aa_column = MiXCRImport.SEQUENCE_NAME_MAP[params.region_type]["AA"]
        nt_column = MiXCRImport.SEQUENCE_NAME_MAP[params.region_type]["NT"]

        ParameterValidator.assert_any_value_present(params.columns_to_load, [aa_column, nt_column], MiXCRImport.__name__, 'columns_to_load')

        if aa_column in params.columns_to_load:
            df["sequence_aa"] = df[aa_column]
        if nt_column in params.columns_to_load:
            df["sequence"] = df[nt_column]

        ImportHelper.junction_to_cdr3(df, params.region_type)
        df.loc[:, "region_type"] = params.region_type.name

        df["duplicate_count"] = df["duplicate_count"].astype(float).astype(int)

        df["v_call"] = MiXCRImport._load_alleles(df, "v_call")
        df["j_call"] = MiXCRImport._load_alleles(df, "j_call")

        ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences)
        ImportHelper.drop_illegal_character_sequences(df, params.import_illegal_characters, params.import_with_stop_codon)
        ImportHelper.load_chains(df)

        return df

    @staticmethod
    def _load_alleles(df: pd.DataFrame, column_name):
        # note: MiXCR omits the '/' for 'TRA.../DV' genes, and remove "*00" for allele if set
        tmp_df = df.apply(lambda row: row[column_name].split(",")[0].split("(")[0].replace("DV", "/DV").replace("//", "/").replace(r'*00', ''), axis=1)
        return tmp_df

    @staticmethod
    def get_documentation():
        doc = str(MiXCRImport.__doc__)

        region_type_values = str([region_type.name for region_type in MiXCRImport.SEQUENCE_NAME_MAP.keys()])[1:-1].replace("'", "`")
        repertoire_fields = list(Repertoire.FIELDS)
        repertoire_fields.remove("region_type")

        mapping = {
            "Valid values for region_type are defined in MiXCRImport.SEQUENCE_NAME_MAP.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
