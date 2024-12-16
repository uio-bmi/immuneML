import json
import logging

import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.SequenceParams import ChainPair, RegionType
from scripts.specification_util import update_docs_per_mapping


class VDJdbImport(DataImport):
    """
    Imports data in VDJdb format into a Repertoire-, Sequence- or ReceptorDataset.
    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
    immune receptors respectively, like antigen specificity.


    **Specification arguments:**

    - path (str): For RepertoireDatasets, this is the path to a directory with VDJdb files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

    - is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

    - metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

    - label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. By default, label_columns for VDJdbImport are [Epitope, Epitope gene, Epitope species]. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

    - paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on the identifier given in the VDJdb column named 'complex.id'.

    - receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as VDJdb uses IMGT junction as CDR3. Specifying any other value will result in importing the sequences as they are. Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

    - column_mapping (dict): A mapping from VDJdb column names to immuneML's internal data representation. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the VDJdb file, or using alternative column names). Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet). For VDJdb, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

                V: v_call
                J: j_call
                CDR3: junction_aa
                complex.id: cell_id
                Gene: locus

    - separator (str): Column separator, for VDJdb this is by default "\\t".


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            datasets:
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
                        import_empty_aa_sequences: False # filter out sequences if they don't have amino acid sequence set
                        # Optional fields with VDJdb-specific defaults, only change when different behavior is required:
                        separator: "\\t" # column separator
                        region_type: IMGT_CDR3 # what part of the sequence to import
                        column_mapping: # column mapping VDJdb: immuneML
                            V: v_call
                            J: j_call
                            CDR3: junction_aa
                            complex.id: sequence_id
                            Gene: chain
                            Epitope: epitope
                            Epitope gene: epitope_gene
                            Epitope species: epitope_species

    """

    KEY_MAPPING = {
        "subject.id": "subject_id"
    }

    def preprocess_file(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sequence_id"] = VDJdbImport.get_sequence_identifiers(df["cell_id"], df["locus"])
        df = self.extract_dict_columns(df)

        return df

    def extract_dict_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        extracts values from meta columns in VDJdb format to separate columns in the data frame, using VDJdbImport.KEY_MAPPING

        Args:
            df: data frame of from file[s] in VDJdb which have already been preprocessed

        Returns:
            the data frame with additional columns where the metadata (if present) were extracted
        """

        for col in ['Meta', 'CDR3fix']:
            if col in df.columns:

                try:

                    meta_df = {}

                    for index, row in df.iterrows():
                        if isinstance(row[col], str):
                            meta = json.loads(row[col])
                            for key, val in meta.items():

                                if isinstance(meta[key], dict):
                                    val_parsed = str(meta[key])
                                else:
                                    val_parsed = val

                                if key in meta_df:
                                    meta_df[key].append(val_parsed)
                                else:
                                    meta_df[key] = ['' for _ in range(index)] + [val_parsed]

                    meta_df = pd.DataFrame(meta_df).astype(str)

                    tmp_col_mapping = {col: col.replace(' ', "_").replace(".", "_") for col in meta_df.columns}
                    meta_df.rename(columns=tmp_col_mapping, inplace=True)

                    df = pd.concat([df, meta_df], axis=1)

                except Exception as e:
                    logging.warning(f"{VDJdbImport.__name__}: an error occurred when parsing the '{col}' field; the "
                                    f"analysis will continue, but none of the information from the '{col}' field will be "
                                    f"available. More details on the error: {e}")

                df.drop(columns=['Meta', 'CDR3fix'], inplace=True, errors='ignore')

        return df

    @staticmethod
    def get_sequence_identifiers(receptor_identifiers, chains):
        receptor_ids_parsed = receptor_identifiers.values.astype(int).astype(str) \
            if receptor_identifiers.dtype == float else receptor_identifiers.values.astype(str)
        sequence_identifiers = pd.Series([el + "_" + chains[i] for i, el in enumerate(receptor_ids_parsed)])
        if sequence_identifiers.is_unique:
            return sequence_identifiers
        else:
            counts = pd.Series(sequence_identifiers).value_counts()
            for id, count in counts[counts > 1].items():
                unique_ids = [f"{id}_{i}" for i in range(1, count + 1)]
                sequence_identifiers.loc[sequence_identifiers == id] = unique_ids
        return sequence_identifiers

    @staticmethod
    def get_documentation():
        doc = str(VDJdbImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet)": f"Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
