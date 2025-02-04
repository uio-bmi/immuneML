import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.SequenceParams import RegionType
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

    - metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the MiXCR files included under the column 'filename' are imported into the RepertoireDataset. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

    - label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence, such as '_', are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as MiXCR format contains the trailing and leading conserved amino acids in the CDR3. Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

    - column_mapping (dict): A mapping from MiXCR column names to immuneML's internal data representation. The columns that specify the sequences to import are handled by the region_type parameter. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the MiXCR file, or using alternative column names). Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet). For MiXCR, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

            cloneCount: duplicate_count
            allVHitsWithScore: v_call
            allJHitsWithScore: j_call
            aaSeqCDR3: junction_aa
            nSeqCDR3: junction
            aaSeqCDR1: cdr1_aa
            nSeqCDR1: cdr1
            aaSeqCDR2: cdr2_aa
            nSeqCDR2: cdr2

    - columns_to_load (list): Specifies which subset of columns must be loaded from the MiXCR file. By default, this is: [cloneCount, allVHitsWithScore, allJHitsWithScore, aaSeqCDR3, nSeqCDR3]

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
                            mixcrColumnName1: metadata_label1
                            mixcrColumnName2: metadata_label2

    """

    def preprocess_file(self, df: pd.DataFrame) -> pd.DataFrame:
        df["v_call"] = load_alleles(df, "v_call")
        df["j_call"] = load_alleles(df, "j_call")

        return df

    @staticmethod
    def get_documentation():
        doc = str(MiXCRImport.__doc__)

        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet)": f"Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc



def load_alleles(df: pd.DataFrame, column_name):
    # note: MiXCR omits the '/' for 'TRA.../DV' genes, and remove "*00" for allele if set
    tmp_df = df.apply(
        lambda row: row[column_name].split(",")[0].split("(")[0].replace("DV", "/DV").replace("//",
                                                                                              "/").replace(
            r'*00', ''), axis=1)
    return tmp_df
