import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.util.ImportHelper import ImportHelper
from scripts.specification_util import update_docs_per_mapping


class TenxGenomicsImport(DataImport):
    """
    Imports data from 10xGenomics into a Repertoire-, Sequence- or ReceptorDataset.
    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
    immune receptors respectively, like antigen specificity.

    The files that should be used as input are named 'Clonotype consensus annotations (CSV)',
    as described here: https://support.10xgenomics.com/single-cell-vdj/software/pipelines/latest/output/annotation#consensus

    Note: by default the 10xGenomics field 'umis' is used to define the immuneML field counts. If you want to use the 10xGenomics
    field reads instead, this can be changed in the column_mapping (set reads: counts).
    Furthermore, the 10xGenomics field clonotype_id is used for the immuneML field cell_id.


    Arguments:

        path (str): This is the path to a directory with 10xGenomics files to import. By default path is set to the current working directory.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or
        ReceptorDataset. By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
        This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
        For setting Sequence- or ReceptorDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

        paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a
        SequenceDataset (paired = False) or a ReceptorDataset (paired = True).
        In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together
        based on the identifier given in the 10xGenomics column named 'clonotype_id'.

        receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor.
        Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.

        import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters
        that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only
        applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal
        characters in the amino acid sequence are removed). By default import_illegal_characters is False.

        import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False.
        By default, import_empty_nt_sequences is set to True.

        import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on
        amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

        region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the
        first and last amino acids are removed from the CDR3 sequence, as 10xGenomics uses IMGT junction as CDR3. Specifying
        any other value will result in importing the sequences as they are.
        Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

        column_mapping (dict): A mapping from 10xGenomics column names to immuneML's internal data representation.
        For 10xGenomics, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

                cdr3: sequence_aas
                cdr3_nt: sequences
                v_gene: v_genes
                j_gene: j_genes
                umis: counts
                chain: chains
                clonotype_id: cell_ids
                consensus_id: sequence_identifiers

        A custom column mapping can be specified here if necessary (for example; adding additional data fields if
        they are present in the 10xGenomics file, or using alternative column names).
        Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS

        metadata_column_mapping (dict): Specifies metadata for Sequence- and ReceptorDatasets. This should specify a
        mapping similar to column_mapping where keys are 10xGenomics column names and values are the names that are internally
        used in immuneML as metadata fields. These metadata fields can be used as prediction labels for Sequence-
        and ReceptorDatasets. For 10xGenomics format, there is no default metadata_column_mapping.
        For setting RepertoireDataset metadata, metadata_column_mapping is ignored, see metadata_file instead.

        separator (str): Column separator, for 10xGenomics this is by default ",".


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_10x_dataset:
            format: 10xGenomics
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                metadata_column_mapping: # metadata column mapping 10xGenomics: immuneML for SequenceDataset
                    tenx_column_name1: metadata_label1
                    tenx_column_name2: metadata_label2
                import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                # Optional fields with 10xGenomics-specific defaults, only change when different behavior is required:
                separator: "," # column separator
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping 10xGenomics: immuneML
                    cdr3: sequence_aas
                    cdr3_nt: sequences
                    v_gene: v_genes
                    j_gene: j_genes
                    umis: counts
                    chain: chains
                    clonotype_id: cell_ids
                    consensus_id: sequence_identifiers

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(TenxGenomicsImport, params, dataset_name)

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        df["frame_types"] = None
        df.loc[df["productive"].eq("True"), "frame_types"] = SequenceFrameType.IN.name

        allowed_productive_values = []
        if params.import_productive:
            allowed_productive_values.append("True")
        if params.import_unproductive:
            allowed_productive_values.append("False")

        df = df[df.productive.isin(allowed_productive_values)]

        ImportHelper.junction_to_cdr3(df, params.region_type)
        ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences)
        ImportHelper.drop_illegal_character_sequences(df, params.import_illegal_characters)
        ImportHelper.update_gene_info(df)

        if "chains" not in df.columns:
            df.loc[:, "chains"] = ImportHelper.load_chains_from_genes(df)

        return df

    @staticmethod
    def import_receptors(df, params):
        df["receptor_identifiers"] = df["cell_ids"]
        return ImportHelper.import_receptors(df, params)

    @staticmethod
    def get_documentation():
        doc = str(TenxGenomicsImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")
        repertoire_fields = list(Repertoire.FIELDS)
        repertoire_fields.remove("region_types")

        mapping = {
            "Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc


