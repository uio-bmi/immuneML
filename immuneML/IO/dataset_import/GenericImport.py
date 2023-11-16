import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset import Dataset
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.util.ImportHelper import ImportHelper
from scripts.specification_util import update_docs_per_mapping


class GenericImport(DataImport):
    """
    Imports data from any tabular file into a Repertoire-, Sequence- or ReceptorDataset. RepertoireDatasets should be
    used when making predictions per repertoire, such as predicting a disease state. SequenceDatasets or ReceptorDatasets
    should be used when predicting values for unpaired (single-chain) and paired immune receptors respectively,
    like antigen specificity.

    This importer works similarly to other importers, but has no predefined default values for which fields are imported,
    and can therefore be tailored to import data from various different tabular files with headers.

    For ReceptorDatasets: this importer assumes the two receptor sequences appear on different lines in the file, and can
    be paired together by a common sequence identifier. If you instead want to import a ReceptorDataset from a tabular
    file that contains both receptor chains on one line, see :ref:`SingleLineReceptor` import


    Specification arguments:

    - path (str): For RepertoireDatasets, this is the path to a directory with files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

    - is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

    - metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. For setting Sequence- or ReceptorDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

    - paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on a common identifier. This identifier should be mapped to the immuneML field 'sequence_identifiers' using the column_mapping.

    - receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values for receptor_chains are the names of the ChainPair enum.

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means immuneML assumes the IMGT junction (including leading C and trailing Y/F amino acids) is used in the input file, and the first and last amino acids will be removed from the sequences to retrieve the IMGT CDR3 sequence. Specifying any other value will result in importing the sequences as they are. Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

    - column_mapping (dict): Required for all datasets. A mapping where the keys are the column names in the input file, and the values correspond to the names used in immuneML's internal data representation. Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS. At least sequences (nucleotide) or sequence_aas (amino acids) must be specified, but all other fields are optional. A column mapping can look for example like this:

        .. indent with spaces
        .. code-block:: yaml

            file_column_amino_acids: sequence_aa
            file_column_v_genes: v_call
            file_column_j_genes: j_call
            file_column_frequencies: duplicate_count

    - column_mapping_synonyms (dict): This is a column mapping that can be used if a column could have alternative names. The formatting is the same as column_mapping. If some columns specified in column_mapping are not found in the file, the columns specified in column_mapping_synonyms are instead attempted to be loaded. For Generic import, there is no default column_mapping_synonyms.

    - metadata_column_mapping (dict): Specifies metadata for Sequence- and ReceptorDatasets. This should specify a mapping similar to column_mapping where keys are file column names and values are the names that are internally used in immuneML as metadata fields. These metadata fields can be used as prediction labels for Sequence- and ReceptorDatasets. This parameter can also be used to specify sequence-level metadata columns for RepertoireDatasets, which can be used by reports. To set prediction label metadata for RepertoireDatasets, see metadata_file instead. There is no default metadata_column_mapping.

        .. indent with spaces
        .. code-block:: yaml

                file_column_antigen_specificity: antigen_specificity

    - columns_to_load (list): Optional; specifies which columns to load from the input file. This may be useful if the input files contain many unused columns. If no value is specified, all columns are loaded.

    - separator (str): Required parameter. Column separator, for example "\\t" or ",". The default value is "\\t"


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_generic_dataset:
            format: Generic
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                separator: "\\t" # column separator
                import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping file: immuneML
                    file_column_amino_acids: sequence_aas
                    file_column_v_genes: v_call
                    file_column_j_genes: j_call
                    file_column_frequencies: duplicate_count
                metadata_column_mapping: # metadata column mapping file: immuneML
                    file_column_antigen_specificity: antigen_specificity
                columns_to_load:  # which subset of columns to load from the file
                    - file_column_amino_acids
                    - file_column_v_genes
                    - file_column_j_genes
                    - file_column_frequencies
                    - file_column_antigen_specificity

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(GenericImport, params, dataset_name)

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences)
        ImportHelper.drop_illegal_character_sequences(df, params.import_illegal_characters, params.import_with_stop_codon)
        ImportHelper.junction_to_cdr3(df, params.region_type)
        df.loc[:, "region_type"] = params.region_type.name
        ImportHelper.load_chains(df)

        return df

    @staticmethod
    def import_receptors(df, params):
        df["receptor_id"] = df["sequence_id"]
        return ImportHelper.import_receptors(df, params)

    @staticmethod
    def get_documentation():
        doc = str(GenericImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")
        repertoire_fields = list(Repertoire.FIELDS)
        repertoire_fields.remove("region_type")

        mapping = {
            "Valid values for receptor_chains are the names of the ChainPair enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
