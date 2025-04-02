import logging
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType, Chain, ChainPair
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType


class ImportHelper:
    DATASET_FORMAT = "yaml"
    INVALID_COLNAME_CHARS = [" ", "#", "&", "."]

    @staticmethod
    def make_new_metadata_file(repertoires: list, metadata: pd.DataFrame, result_path: Path, dataset_name: str) -> Path:
        new_metadata = metadata.copy()
        new_metadata.loc[:, "filename"] = [repertoire.data_filename.name for repertoire in repertoires]
        new_metadata.loc[:, "identifier"] = [repertoire.identifier for repertoire in repertoires]

        metadata_filename = result_path / f"{dataset_name}_metadata.csv"
        new_metadata.to_csv(metadata_filename, index=False, sep=",")

        return metadata_filename

    @staticmethod
    def get_repertoire_filename_from_metadata_row(metadata_row: pd.Series, params: DatasetImportParams) -> Path:
        if params.path is None:
            params.path = params.dataset_file.parent if params.dataset_file else params.metadata_file.parent
        filename = params.path / f"{metadata_row['filename']}"
        if not filename.is_file():
            filename = params.path / f"repertoires/{metadata_row['filename']}"
        return filename

    @staticmethod
    def get_receptor_filter_sort_kwargs(df: pd.DataFrame, warning=""):
        kwargs = {"by": ["cell_id", "locus"], "ascending": [True, True]}

        # umi's in single-cells sequencing data are the most trustworthy indicator of what the 'true' sequences were
        # duplicate count is less accurate but better than nothing
        if "umi_count" in df.columns and len(set(df["umi_count"])) > 1:
            warning += "Attempting to select the top chain pair based on highest umi_count"
            kwargs["by"].append("umi_count")
            kwargs["ascending"].append(False)

            # consensus_count reflects the count per umi; only useful if there are umi count ties
            if "consensus_count" in df.columns and len(set(df["consensus_count"])) > 1:
                warning += " and consensus_count"
                kwargs["by"].append("consensus_count")
                kwargs["ascending"].append(False)

        elif "duplicate_count" in df.columns and len(set(df["duplicate_count"])) > 1:
            warning += ("Since umi_count was not set, attempting to select the top chain pair based on highest "
                        "duplicate_count")
            kwargs["by"].append("duplicate_count")
            kwargs["ascending"].append(False)
        else:
            warning += ("Since duplicate_count or umi_count was not set, two random chains will be selected for "
                        "each receptor if more than 2 chains are present.")
        logging.warning(warning)

        return kwargs

    @staticmethod
    def remove_illegal_chains(df: pd.DataFrame, chain_pair: ChainPair = None) -> pd.DataFrame:
        if chain_pair is not None:
            rows_with_allowed_chains = [ChainPair.is_allowed(Chain.get_chain(item), chain_pair) for item in df.locus]
            invalid_sequences = len(rows_with_allowed_chains) - sum(rows_with_allowed_chains)
            if invalid_sequences != 0:
                logging.warning(f"Some sequences in the file have locus that is not in {chain_pair}. "
                                f"Removing {invalid_sequences} sequences.")

                df = df.loc[rows_with_allowed_chains]

        return df

    @staticmethod
    def filter_illegal_receptors(df: pd.DataFrame, chain_pair: ChainPair = None) -> pd.DataFrame:
        df = ImportHelper.remove_illegal_chains(df, chain_pair)

        assert "cell_id" in df.columns, "Receptor datasets cannot be constructed if cell_id field is missing."

        cell_id_counts = df.groupby('cell_id').size()

        logging.info(f"Total number of unique cell_ids (potential receptors): {len(cell_id_counts)}")

        if sum(cell_id_counts > 2) > 0:
            kwargs = ImportHelper.get_receptor_filter_sort_kwargs(df, f"Found {sum(cell_id_counts > 2)} "
                                                                      f"receptors with > 2 cell_ids. ")

            assert "locus" in df.columns, "Receptor datasets cannot be constructed if locus field is missing."

            df.sort_values(**kwargs, inplace=True)
            df.drop_duplicates(subset=["cell_id", "locus"], keep="first", inplace=True)

            cell_id_counts = df.groupby('cell_id').size()

        if sum(cell_id_counts == 1) > 0:
            logging.warning(f"Found {sum(cell_id_counts == 1)} receptors with 1 cell_id, removing these receptors.")
            df = df.loc[df.cell_id.isin(cell_id_counts[cell_id_counts >= 2].index)]

        logging.info(f"Total number of unique cell_ids (receptors) left after filtering: {df['cell_id'].nunique()}")

        return df

    @staticmethod
    def add_default_fields_for_airr_seq_set(df: pd.DataFrame):
        default_fields = {f.name: f.type for f in fields(AIRRSequenceSet) if f.name not in df.columns}
        fields_dict = {}
        for f_name, f_type in default_fields.items():
            val = AIRRSequenceSet.get_neutral_value(f_type)
            fields_dict[f_name] = [val for _ in range(df.shape[0])]

        df = pd.concat([df.reset_index().drop(columns='index'), pd.DataFrame(fields_dict)], axis=1)

        return df

    @staticmethod
    def get_standardized_name(column_name: str) -> str:
        for invalid_char in ImportHelper.INVALID_COLNAME_CHARS:
            column_name = column_name.replace(invalid_char, "_")
        return column_name

    @staticmethod
    def standardize_column_names(df):
        invalid_col_names = {col: ImportHelper.get_standardized_name(col)
                             for col in df.columns if any(el in col for el in ImportHelper.INVALID_COLNAME_CHARS)}
        if len(invalid_col_names.keys()) > 0:
            logging.warning(
                f"Note that column names that contain characters which are not letters, numbers nor '_' signs"
                f" have been renamed to replace these invalid characters with '_' instead: {invalid_col_names}")

        return df.rename(columns=invalid_col_names)

    @staticmethod
    def extract_locus_from_data(df: pd.DataFrame, params: DatasetImportParams, dataset_name: str):
        if 'locus' not in df.columns or any(df['locus'] == ''):
            if 'v_call' in df.columns and all(df.v_call != ''):
                locus_list = [v_call[:3] for v_call in df.v_call]
            elif 'j_call' in df.columns and all(df.j_call != ''):
                locus_list = [j_call[:3] for j_call in df.j_call]
            else:
                logging.info(f"{ImportHelper.__name__}: locus could not be extracted for dataset {dataset_name}.")
                return df
            df['locus'] = [Chain.get_chain_value(item) for item in locus_list]
        return df

    @staticmethod
    def standardize_none_values(dataframe: pd.DataFrame):
        return (dataframe.replace(
            {key: Constants.UNKNOWN for key in ["unresolved", "no data", "na", "unknown", 'nan']})
                .replace(np.nan, AIRRSequenceSet.get_neutral_value(float)))

    @staticmethod
    def standardize_bool_values(dataframe: pd.DataFrame):
        return dataframe.replace({True: 'T', False: 'F', 'TRUE': 'T', 'FALSE': 'F',
                                  'True': 'T', 'False': 'F', 'true': 'T', 'false': 'F'})

    @staticmethod
    def add_cdr3_from_junction(df: pd.DataFrame):
        if 'junction' in df.columns and ('cdr3' not in df.columns or all(df['cdr3'] == '')):
            df['cdr3'] = df.junction.str[3:-3]
        if 'junction_aa' in df.columns and ('cdr3_aa' not in df.columns or all(df['cdr3_aa'] == '')):
            df['cdr3_aa'] = df.junction_aa.str[1:-1]
        return df

    @staticmethod
    def drop_empty_sequences(dataframe: pd.DataFrame, import_empty_aa_sequences: bool,
                             import_empty_nt_sequences: bool, region_type: RegionType) -> pd.DataFrame:
        sequence_types = []

        if not import_empty_aa_sequences:
            sequence_types.append(SequenceType.AMINO_ACID)
        if not import_empty_nt_sequences:
            sequence_types.append(SequenceType.NUCLEOTIDE)

        for sequence_type in sequence_types:
            sequence_colname = region_type.value if sequence_type == SequenceType.NUCLEOTIDE else region_type.value + "_aa"
            sequence_name = sequence_type.name.lower().replace("_", " ")

            if sequence_colname in dataframe.columns:
                try:
                    empty = dataframe[sequence_colname].isnull() | (dataframe[sequence_colname] == '')
                    n_empty = sum(empty)
                except Exception as e:
                    raise e

                if n_empty > 0:
                    dataframe = dataframe.loc[~empty]
                    logging.warning(f"{ImportHelper.__name__}: {n_empty} sequences were removed from the dataset "
                                    f"because they contained an empty {sequence_name} sequence after preprocessing. ")
            else:
                logging.warning(f"{ImportHelper.__name__}: column {sequence_colname} was not set, but is required "
                                f"for filtering. Skipping this filtering...")

        return dataframe

    @staticmethod
    def drop_illegal_character_sequences(dataframe: pd.DataFrame, import_illegal_characters: bool,
                                         import_with_stop_codon: bool, region_type: RegionType) -> pd.DataFrame:
        for sequence_type in SequenceType:
            if not import_illegal_characters:
                sequence_name = sequence_type.name.lower().replace("_", " ")

                legal_alphabet = EnvironmentSettings.get_sequence_alphabet(sequence_type)
                if sequence_type == SequenceType.AMINO_ACID and import_with_stop_codon:
                    legal_alphabet.append(Constants.STOP_CODON)

                sequence_col_name = region_type.value if sequence_type == SequenceType.NUCLEOTIDE else region_type.value + "_aa"

                if sequence_col_name in dataframe.columns:
                    is_illegal_seq = [ImportHelper.is_illegal_sequence(sequence, legal_alphabet) for sequence in
                                      dataframe[sequence_col_name]]
                    n_illegal = sum(is_illegal_seq)
                    n_total = dataframe.shape[0]

                    if n_illegal > 0:
                        dataframe.drop(dataframe.loc[is_illegal_seq].index, inplace=True)
                        logging.warning(
                            f"{ImportHelper.__name__}: {n_illegal}/{n_total} sequences were removed from the dataset "
                            f"because their {sequence_name} sequence contained illegal characters. ")

                else:
                    logging.warning(f"{ImportHelper.__name__}: column {sequence_col_name} is missing, illegal "
                                    f"characters were not checked.")

        return dataframe

    @staticmethod
    def is_illegal_sequence(sequence, legal_alphabet) -> bool:
        if sequence is None:
            return False
        elif not isinstance(sequence, str):
            return True
        else:
            return not all(character in legal_alphabet for character in sequence)

    @staticmethod
    def get_sequence_filenames(path: Path, dataset_name: str):
        data_file_extensions = ("*.tsv", "*.csv", "*.txt")

        if path.is_file():
            filenames = [path]
        elif path.is_dir():
            filenames = []

            for pattern in data_file_extensions:
                filenames.extend(list(path.glob(pattern)))
        else:
            raise ValueError(f"ImportHelper: path '{path}' given in YAML specification is not a valid path. "
                             f"This parameter can either point to a single file with immune receptor data or to a "
                             f"directory containing such files.")

        assert len(
            filenames) >= 1, f"ImportHelper: the dataset {dataset_name} cannot be imported, no files were found under {path}.\n" \
                             f"Note that only files with the following extensions can be imported: {data_file_extensions}"
        return filenames

    @staticmethod
    def extract_sequence_dataset_params(items=None, params=None) -> dict:
        result = {}
        if params is not None:
            result = {'region_type': params.region_type, 'receptor_chains': params.receptor_chains,
                      'organism': params.organism}
        if items is not None:
            for index, item in enumerate(items):
                metadata = item.metadata if params.paired else item.metadata.custom_params if item.metadata is not None else {}
                for key in metadata:
                    if key in result and isinstance(result[key], set):
                        result[key].add(metadata[key])
                    elif key not in result:
                        result[key] = {metadata[key]}
        return result

    @classmethod
    def filter_illegal_sequences(cls, df: pd.DataFrame, params: DatasetImportParams, location: str):
        try:
            if params.import_productive and hasattr(df, "productive"):
                if set(df.productive).issubset({'T', 'F'}):
                    logging.info(f"ImportHelper: filtering {len(df)} sequences to only keep productive == 'T'...")
                    df = df.loc[df.productive == 'T']
                    logging.info(f"ImportHelper: ... {len(df)} sequences left after filtering")
                else:
                    logging.warning(f"ImportHelper: import_productive was set to True, but unexpected values were "
                                    f"found in the 'productive' column: {set(df.productive)} (expected 'T', 'F'). "
                                    f"This filtering has been skipped.")

        except AttributeError as e:
            logging.warning(f"An error occurred while filtering unproductive sequences while importing the "
                            f"dataset {location}. Error: {e}. Filtering will be skipped.")

        try:
            if not params.import_out_of_frame and hasattr(df, "vj_in_frame"):

                if set(df.vj_in_frame).issubset({'T', 'F'}):
                    logging.info(f"ImportHelper: import_out_of_frame is set to False. Filtering {len(df)} sequences "
                                 f"to only keep vj_in_frame != 'F'...")
                    df = df.loc[df.vj_in_frame != 'F']
                else:
                    logging.warning(f"ImportHelper: import_out_of_frame was set to False, but unexpected values were "
                                    f"found in the 'vj_in_frame' column: {set(df.vj_in_frame)} (expected 'T', 'F'). "
                                    f"This filtering has been skipped.")
        except AttributeError as e:
            logging.warning(f"An error occurred while filtering out-of-frame sequences while importing the "
                            f"dataset {location}. Error: {e}. Filtering will be skipped.")

        return df
