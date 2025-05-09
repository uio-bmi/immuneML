import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.SequenceParams import RegionType
from immuneML.util.AdaptiveImportHelper import AdaptiveImportHelper
from scripts.specification_util import update_docs_per_mapping


class ImmunoSEQSampleImport(DataImport):
    """
    Imports data from Adaptive Biotechnologies immunoSEQ Analyzer sample-level .tsv files into a
    Repertoire-, or SequenceDataset.
    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
    antigen specificity.

    The format of the files imported by this importer is described here in section 3.4.13
    https://clients.adaptivebiotech.com/assets/downloads/immunoSEQ_AnalyzerManual.pdf
    Alternatively, to import rearrangement-level .tsv files, see :ref:`ImmunoSEQRearrangement` import.
    The only difference between these two importers is which columns they load from the .tsv files.


    **Specification arguments:**

    - path (str): For RepertoireDatasets, this is the path to a directory with files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

    - is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset. By default, is_repertoire is set to True.

    - metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the files included under the column 'filename' are imported into the RepertoireDataset. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

    - label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

    - import_productive (bool): Whether productive sequences (with value 'In' in column frame_type) should be included in the imported sequences. By default, import_productive is True.

    - import_with_stop_codon (bool): Whether sequences with stop codons (with value 'Stop' in column frame_type) should be included in the imported sequences. By default, import_with_stop_codon is False.

    - import_out_of_frame (bool): Whether out of frame sequences (with value 'Out' in column frame_type) should be included in the imported sequences. By default, import_out_of_frame is False.

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as immunoSEQ files use the IMGT junction. Specifying any other value will result in importing the sequences as they are. Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

    - column_mapping (dict): A mapping from immunoSEQ column names to immuneML's internal data representation. For immunoSEQ sample-level files, this is by default set to the values shown bellow in YAML format.         A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the file, or using alternative column names). Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet). For ImmunoSEQ sample import, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

              nucleotide: sequence
              aminoAcid: junction_aa
              vGeneName: v_call
              jGeneName: j_call
              sequenceStatus: frame_type
              count (templates/reads): duplicate_count

    - column_mapping_synonyms (dict): This is a column mapping that can be used if a column could have alternative names. The formatting is the same as column_mapping. If some columns specified in column_mapping are not found in the file, the columns specified in column_mapping_synonyms are instead attempted to be loaded. For immunoSEQ sample .tsv files, there is no default column_mapping_synonyms.

    - columns_to_load (list): Specifies which subset of columns must be loaded from the file. By default, this is: [nucleotide, aminoAcid, count (templates/reads), vFamilyName, vGeneName, vGeneAllele, jFamilyName, jGeneName, jGeneAllele, sequenceStatus]; these are the columns from the original file that will be imported

    - metadata_column_mapping (dict): Specifies metadata for Sequence- and ReceptorDatasets. This should specify a mapping similar to column_mapping where keys are immunoSEQ column names and values are the names that are internally used in immuneML as metadata fields. These metadata fields can be used as prediction labels for Sequence- and ReceptorDatasets. This parameter can also be used to specify sequence-level metadata columns for RepertoireDatasets, which can be used by reports. To set prediction label metadata for RepertoireDatasets, see metadata_file instead. For immunoSEQ sample .tsv files, there is no default metadata_column_mapping.

    - separator (str): Column separator, for ImmunoSEQ files this is by default "\\t".


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            datasets:
                my_immunoseq_dataset:
                    format: ImmunoSEQSample
                    params:
                        path: path/to/files/
                        is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                        metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                        metadata_column_mapping: # metadata column mapping ImmunoSEQ: immuneML for SequenceDataset
                            immunoseq_column_name1: metadata_label1
                            immunoseq_column_name2: metadata_label2
                        import_productive: True # whether to include productive sequences in the dataset
                        import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                        import_out_of_frame: False # whether to include out of frame sequences in the dataset
                        import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                        import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                        import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                        # Optional fields with ImmunoSEQ sample-specific defaults, only change when different behavior is required:
                        separator: "\\t" # column separator
                        columns_to_load: # subset of columns to load
                        - nucleotide
                        - aminoAcid
                        - count (templates/reads)
                        - vFamilyName
                        - vGeneName
                        - vGeneAllele
                        - jFamilyName
                        - jGeneName
                        - jGeneAllele
                        - sequenceStatus
                        region_type: IMGT_CDR3 # what part of the sequence to import
                        column_mapping: # column mapping immunoSEQ: immuneML
                            nucleotide: sequence
                            aminoAcid: junction_aa
                            vGeneName: v_call
                            jGeneName: j_call
                            sequenceStatus: frame_type
                            vFamilyName: v_family
                            jFamilyName: j_family
                            vGeneAllele: v_allele
                            jGeneAllele: j_allele
                            count (templates/reads): duplicate_count

    """

    def preprocess_file(self, df: pd.DataFrame) -> pd.DataFrame:
        return AdaptiveImportHelper.preprocess_dataframe(df, self.params)

    @staticmethod
    def get_documentation():
        doc = str(ImmunoSEQSampleImport.__doc__)

        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet)": f"Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
