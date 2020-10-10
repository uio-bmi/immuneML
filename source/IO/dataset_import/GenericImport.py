import pandas as pd

from scripts.specification_util import update_docs_per_mapping
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset import Dataset
from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.RegionType import RegionType
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.ImportHelper import ImportHelper


class GenericImport(DataImport):
    """
    Imports data from any tabular file into a Repertoire-, Sequence- or ReceptorDataset. RepertoireDatasets should be
    used when making predictions per repertoire, such as predicting a disease state. SequenceDatasets or ReceptorDatasets
    should be used when predicting values for for unpaired (single-chain) and paired immune receptors respectively,
    like antigen specificity.

    This importer works similarly to other importers, but has no predefined default values for which fields are imported,
    and can therefore be tailored to import data from various different tabular files with headers.

    For ReceptorDatasets: this importer assumes the two receptor sequences appear on different lines in the file, and can
    be paired together by a common sequence identifier. If you instead want to import a ReceptorDataset from a tabular
    file that contains both receptor chains on one line, see SingleLineReceptorImport.


    Arguments:
        path (str): Required parameter. This is the path to a directory with files to import.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or
            ReceptorDataset. By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
            This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
            For setting Sequence- or ReceptorDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

        paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a
            SequenceDataset (paired = False) or a ReceptorDataset (paired = True).
            In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based
            on a common identifier. This identifier should be mapped to the immuneML field 'sequence_identifiers' using
            the column_mapping.

        receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor.
            Valid values for receptor_chains are the names of the ChainPair enum.

        region_type (str): Which part of the sequence to import. When IMGT_CDR3 is specified, immuneML assumes the IMGT
            junction (including leading C and trailing Y/F amino acids) is used in the input file, and the first and last
            amino acids will be removed from the sequences to retrieve the IMGT CDR3 sequence. Specifying any other value
            will result in no trimming of the imported sequences.
            Valid values for region_type are the names of the RegionType enum.

        column_mapping (dict): Required for all datasets. A mapping where the keys are the column names in the input file,
            and the values correspond to the names used in immuneML's internal data representation.
            Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS
            At least sequences (nucleotide) or sequence_aas (amino acids) must be specified, but all other fields
            are optional. A column mapping can look for example like this:
                file_column_amino_acids: sequence_aas
                file_column_v_genes: v_genes
                file_column_j_genes: j_genes
                file_column_frequencies: counts

        metadata_column_mapping (dict): Optional; specifies metadata for Sequence- and ReceptorDatasets. This is a column
            mapping that is formatted similarly to column_mapping, but here the values are the names that immuneML internally
            uses as metadata fields. These fields can subsequently be used as labels in instructions (for example labels
            that are used for prediction by ML methods). This column mapping could for example look like this:
                file_column_antigen_specificity: antigen_specificity
            The label antigen_specificity can now be used throughout immuneML.
            For setting RepertoireDataset metadata, metadata_column_mapping is ignored, see metadata_file instead.

        separator (str): Column separator, for example "\\t" or ",".


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_vdjdb_dataset:
            format: Generic
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                separator: "\\t" # column separator
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping file: immuneML
                    file_column_amino_acids: sequence_aas
                    file_column_v_genes: v_genes
                    file_column_j_genes: j_genes
                    file_column_frequencies: counts
                metadata_column_mapping: # metadata column mapping file: immuneML
                    file_column_antigen_specificity: antigen_specificity
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(GenericImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        ImportHelper.junction_to_cdr3(df, params.region_type)
        return df

    @staticmethod
    def import_receptors(df, params):
        df["receptor_identifiers"] = df["sequence_identifiers"]
        return ImportHelper.import_receptors(df, params)

    @staticmethod
    def get_documentation():
        doc = str(GenericImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")
        repertoire_fields = Repertoire.FIELDS
        repertoire_fields.remove("region_type")

        mapping = {
            "Valid values for receptor_chains are the names of the ChainPair enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the RegionType enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc

