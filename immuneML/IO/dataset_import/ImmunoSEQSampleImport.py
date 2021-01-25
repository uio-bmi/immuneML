import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.util.AdaptiveImportHelper import AdaptiveImportHelper
from immuneML.util.ImportHelper import ImportHelper
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


    Arguments:

        path (str): This is the path to a directory with files to import. By default path is set to the current working directory.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset.
        By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
        This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
        Only the files included under the column 'filename' are imported into the RepertoireDataset.
        For setting SequenceDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

        import_productive (bool): Whether productive sequences (with value 'In' in column frame_type) should be included
        in the imported sequences. By default, import_productive is True.

        import_with_stop_codon (bool): Whether sequences with stop codons (with value 'Stop' in column frame_type) should
        be included in the imported sequences. By default, import_with_stop_codon is False.

        import_out_of_frame (bool): Whether out of frame sequences (with value 'Out' in column frame_type) should
        be included in the imported sequences. By default, import_out_of_frame is False.

        import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters
        that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only
        applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal
        characters in the amino acid sequence are removed). By default import_illegal_characters is False.

        import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False.
        By default, import_empty_nt_sequences is set to True.

        import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on
        amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

        region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the
        first and last amino acids are removed from the CDR3 sequence, as immunoSEQ files use the IMGT junction.
        Specifying any other value will result in importing the sequences as they are.
        Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

        column_mapping (dict): A mapping from immunoSEQ column names to immuneML's internal data representation.
        For immunoSEQ sample-level files, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

                nucleotide: sequences
                aminoAcid: sequence_aas
                vGeneName: v_genes
                jGeneName: j_genes
                sequenceStatus: frame_types
                vFamilyName: v_subgroups
                jFamilyName: j_subgroups
                vGeneAllele: v_alleles
                jGeneAllele: j_alleles
                count (templates/reads): counts

        A custom column mapping can be specified here if necessary (for example; adding additional data fields if
        they are present in the file, or using alternative column names).
        Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS

        columns_to_load (list): Specifies which subset of columns must be loaded from the file. By default, this is:
        [nucleotide, aminoAcid, count (templates/reads), vFamilyName, vGeneName, vGeneAllele, jFamilyName, jGeneName, jGeneAllele, sequenceStatus] # columns from the original file that will be imported

        metadata_column_mapping (dict): Specifies metadata for SequenceDatasets. This should specify a mapping similar
        to column_mapping where keys are immunoSEQ column names and values are the names that are internally used in immuneML
        as metadata fields. These metadata fields can be used as prediction labels for SequenceDatasets.
        For immunoSEQ .tsv files, there is no default metadata_column_mapping.
        For setting RepertoireDataset metadata, metadata_column_mapping is ignored, see metadata_file instead.

        separator (str): Column separator, for ImmunoSEQ files this is by default "\\t".


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

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
                    nucleotide: sequences
                    aminoAcid: sequence_aas
                    vGeneName: v_genes
                    jGeneName: j_genes
                    sequenceStatus: frame_types
                    vFamilyName: v_subgroups
                    jFamilyName: j_subgroups
                    vGeneAllele: v_alleles
                    jGeneAllele: j_alleles
                    count (templates/reads): counts
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(ImmunoSEQSampleImport, params, dataset_name)

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        return AdaptiveImportHelper.preprocess_dataframe(df, params)

    @staticmethod
    def get_documentation():
        doc = str(ImmunoSEQSampleImport.__doc__)

        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")
        repertoire_fields = list(Repertoire.FIELDS)
        repertoire_fields.remove("region_types")

        mapping = {
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
