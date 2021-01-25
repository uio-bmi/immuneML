import json
import warnings

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


class VDJdbImport(DataImport):
    """
    Imports data in VDJdb format into a Repertoire-, Sequence- or ReceptorDataset.
    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
    immune receptors respectively, like antigen specificity.


    Arguments:

        path (str): This is the path to a directory with VDJdb files to import. By default path is set to the current working directory.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or
        ReceptorDataset. By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
        This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
        For setting Sequence- or ReceptorDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

        paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a
        SequenceDataset (paired = False) or a ReceptorDataset (paired = True).
        In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together
        based on the identifier given in the VDJdb column named 'complex.id'.

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
        first and last amino acids are removed from the CDR3 sequence, as VDJdb uses IMGT junction as CDR3. Specifying
        any other value will result in importing the sequences as they are.
        Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

        column_mapping (dict): A mapping from VDJdb column names to immuneML's internal data representation.
        For VDJdb, this is by default set to:

            .. indent with spaces
            .. code-block:: yaml

                    V: v_genes
                    J: j_genes
                    CDR3: sequence_aas
                    complex.id: sequence_identifiers
                    Gene: chains

        A custom column mapping can be specified here if necessary (for example; adding additional data fields if
        they are present in the VDJdb file, or using alternative column names).
        Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS

        metadata_column_mapping (dict): Specifies metadata for Sequence- and ReceptorDatasets. This should specify
        a mapping where keys are VDJdb column names and values are the names that are internally used in immuneML
        as metadata fields.
        For VDJdb format, this parameter is by default set to:

            .. indent with spaces
            .. code-block:: yaml

                    Epitope: epitope
                    Epitope gene: epitope_gene
                    Epitope species: epitope_species

        This means that epitope, epitope_gene and epitope_species can be specified as prediction labels for
        Sequence- and ReceptorDatasets. Custom metadata labels can be defined here as well.
        For setting RepertoireDataset metadata, metadata_column_mapping is ignored, see metadata_file instead.

        separator (str): Column separator, for VDJdb this is by default "\\t".


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_vdjdb_dataset:
            format: VDJdb
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                # Optional fields with VDJdb-specific defaults, only change when different behavior is required:
                separator: "\\t" # column separator
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping VDJdb: immuneML
                    V: v_genes
                    J: j_genes
                    CDR3: sequence_aas
                    complex.id: sequence_identifiers
                    Gene: chains
                metadata_column_mapping: # metadata column mapping VDJdb: immuneML
                    Epitope: epitope
                    Epitope gene: epitope_gene
                    Epitope species: epitope_species

    """

    KEY_MAPPING = {
        "subject.id": "subject_id"
    }

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(VDJdbImport, params, dataset_name)

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        df["frame_types"] = SequenceFrameType.IN.name
        ImportHelper.junction_to_cdr3(df, params.region_type)

        if not params.is_repertoire and params.paired:
            n_single_chains = sum(df["sequence_identifiers"] == "0")
            if n_single_chains > 0:
                df.drop(df.loc[df["sequence_identifiers"] == "0"].index, inplace=True)
                warnings.warn(f"VDJdbImport: {n_single_chains} single chains were removed when trying to create a ReceptorDataset.\n"
                              f"To import all chains as a SequenceDataset, use paired = False")
        else:
            df.loc[df["sequence_identifiers"] == "0", "sequence_identifiers"] = None

        if "chains" not in df.columns:
            df.loc[:, "chains"] = ImportHelper.load_chains_from_genes(df)

        df["receptor_identifiers"] = df["sequence_identifiers"]
        df["sequence_identifiers"] = VDJdbImport.get_sequence_identifiers(df["sequence_identifiers"], df["chains"])

        ImportHelper.update_gene_info(df)
        ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences)
        ImportHelper.drop_illegal_character_sequences(df, params.import_illegal_characters)
        df = VDJdbImport.extract_meta_columns(df, params)

        return df

    @staticmethod
    def extract_meta_columns(df: pd.DataFrame, params) -> pd.DataFrame:
        """
        extracts values from meta columns in VDJdb format to separate columns in the data frame, using VDJdbImport.KEY_MAPPING

        Note: the KEY_MAPPING values cannot be manually set, but correspond to the standard names as defined by the AIRR Community

        Args:
            df: data frame of from file[s] in VDJdb which have already been preprocessed
            params: parameters from importing dataset which includes the information on metadata column mapping

        Returns:
            the data frame with additional columns where the metadata (if present) were extracted
        """
        for key, new_key in VDJdbImport.KEY_MAPPING.items():
            df.loc[:, new_key] = ""
        meta_name = params.metadata_column_mapping["Meta"] if params.metadata_column_mapping is not None and "Meta" in params.metadata_column_mapping else "Meta"
        if meta_name in df.columns:
            for index, row in df.iterrows():
                if isinstance(row[meta_name], str):
                    meta = json.loads(row[meta_name])
                    for key, new_key in VDJdbImport.KEY_MAPPING.items():
                        if key in meta:
                            row[new_key] = meta[key]

        return df

    @staticmethod
    def get_sequence_identifiers(receptor_identifiers, chains):
        sequence_identifiers = receptor_identifiers + "_" + chains
        if sequence_identifiers.is_unique:
            return sequence_identifiers
        else:
            counts = sequence_identifiers.value_counts()
            for id, count in counts[counts > 1].iteritems():
                unique_ids = [f"{id}{i}" for i in range(1, count+1)]
                sequence_identifiers.loc[sequence_identifiers == id] = unique_ids
        return sequence_identifiers

    @staticmethod
    def import_receptors(df, params):
        return ImportHelper.import_receptors(df, params)

    @staticmethod
    def get_documentation():
        doc = str(VDJdbImport.__doc__)

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



