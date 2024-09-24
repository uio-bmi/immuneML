import logging
import logging
import warnings
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


class ImportHelper:
    DATASET_FORMAT = "yaml"


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
        filename = params.path / f"{metadata_row['filename']}"
        if not filename.is_file():
            filename = params.path / f"repertoires/{metadata_row['filename']}"
        return filename


    @staticmethod
    def load_sequence_dataframe(filepath, params, alternative_load_func=None):
        try:
            if alternative_load_func:
                df = alternative_load_func(filepath, params)
            else:
                df = ImportHelper.safe_load_dataframe(filepath, params)
        except Exception as ex:
            raise Exception(
                f"{ex}\n\nImportHelper: an error occurred during dataset import while parsing the input file: {filepath}.\n"
                f"Please make sure this is a correct immune receptor data file (not metadata).\n"
                f"The parameters used for import are {params}.\nFor technical description of the error, see the log above. "
                f"For details on how to specify the dataset import, see the documentation.").with_traceback(
                ex.__traceback__)

        ImportHelper.rename_dataframe_columns(df, params)
        ImportHelper.standardize_none_values(df)

        return df

    @staticmethod
    def filter_illegal_receptors(df: pd.DataFrame) -> pd.DataFrame:
        assert "cell_id" in df.columns, "Receptor datasets cannot be constructed if cell_id field is missing."

        cell_id_counts = df.groupby('cell_id').size()

        if not (cell_id_counts == 2).all():
            logging.warning("There are cells in the dataset that don't have exactly two chains. "
                            "Those will be filtered out from the dataset.")

        return df.loc[cell_id_counts == 2, :]

    @staticmethod
    def parse_sequence_dataframe(df: pd.DataFrame, params: DatasetImportParams, dataset_name: str) -> pd.DataFrame:
        if hasattr(params, "column_mapping") and params.column_mapping is not None:
            df.rename(columns=params.column_mapping, inplace=True)

        df = ImportHelper.standardize_column_names(df)
        df = ImportHelper.standardize_none_values(df)
        df = ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences,
                                               params.region_type)
        df = ImportHelper.drop_illegal_character_sequences(df, params.import_illegal_characters,
                                                           params.import_with_stop_codon, params.region_type)
        df = ImportHelper.filter_illegal_sequences(df, params, dataset_name)

        df = ImportHelper.add_default_fields_for_airr_seq_set(df)

        return df

    @staticmethod
    def add_default_fields_for_airr_seq_set(df: pd.DataFrame):
        default_fields = {f.name: f.type for f in fields(AIRRSequenceSet) if f.name not in df.columns}
        fields_dict = {}
        for f_name, f_type in default_fields.items():
            val = AIRRSequenceSet.get_neutral_value(f_type)
            fields_dict[f_name] = [val for _ in range(df.shape[0])]

        df = pd.concat([df, pd.DataFrame(fields_dict)], axis=1)

        return df


    @staticmethod
    def standardize_column_names(df):
        invalid_chars = [" ", "#", "&"]
        invalid_col_names = {col: col.replace(" ", "_").replace("#", "_").replace("&", "_")
                             for col in df.columns if any(el in col for el in invalid_chars)}
        if len(invalid_col_names.keys()) > 0:
            logging.warning(
                f"Note that column names that contain characters which are not letters, numbers nor '_' signs"
                f" have been renamed to replace these invalid characters with '_' instead: {invalid_col_names}")

        return df.rename(columns=invalid_col_names)

    @staticmethod
    def standardize_none_values(dataframe: pd.DataFrame):
        return dataframe.replace(
            {key: Constants.UNKNOWN for key in ["unresolved", "no data", "na", "unknown"]})

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
                n_empty = sum(dataframe[sequence_colname].isnull())
                if n_empty > 0:
                    dataframe.drop(dataframe.loc[dataframe[sequence_colname].isnull()].index, inplace=True)
                    warnings.warn(
                        f"{ImportHelper.__name__}: {n_empty} sequences were removed from the dataset because they contained an empty {sequence_name} "
                        f"sequence after preprocessing. ")
            else:
                warnings.warn(
                    f"{ImportHelper.__name__}: column {sequence_colname} was not set, but is required for filtering. Skipping this filtering...")

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
        else:
            return not all(character in legal_alphabet for character in sequence)

    @staticmethod
    def load_chains(df: pd.DataFrame):
        if "locus" in df.columns:
            df["locus"] = ImportHelper.load_chains_from_chains(df)
        else:
            df["locus"] = ImportHelper.load_chains_from_genes(df)

    @staticmethod
    def load_chains_from_chains(df: pd.DataFrame) -> list:
        return [Chain.get_chain(chain_str).value if chain_str is not None else None for chain_str in df["locus"]]

    @staticmethod
    def load_chains_from_genes(df: pd.DataFrame) -> list:
        return df.apply(ImportHelper.get_chain_for_row, axis=1)

    @staticmethod
    def get_chain_for_row(row):
        for col in ["v_call", "j_call"]:
            if col in row and row[col] is not None:
                chain = Chain.get_chain(str(row[col])[0:3])
                return chain.value if chain is not None else None
        return None

    @staticmethod
    def junction_to_cdr3(df: pd.DataFrame, region_type: RegionType):
        """
        If RegionType is CDR3, the leading C and trailing W are removed from the sequence to match the IMGT CDR3 definition.
        This method alters the data in the provided dataframe.
        """

        if region_type == RegionType.IMGT_CDR3:
            if "sequence_aa" in df:
                df.loc[:, "sequence_aa"] = df["sequence_aa"].str[1:-1]
            if "sequence" in df:
                df.loc[:, "sequence"] = df["sequence"].str[3:-3]
            df["region_type"] = region_type.name

    @staticmethod
    def strip_alleles(df: pd.DataFrame, column_name):
        return ImportHelper.strip_suffix(df, column_name, Constants.ALLELE_DELIMITER)

    @staticmethod
    def strip_genes(df: pd.DataFrame, column_name):
        return ImportHelper.strip_suffix(df, column_name, Constants.GENE_DELIMITER)

    @staticmethod
    def strip_suffix(df: pd.DataFrame, column_name, delimiter):
        """
        Safely removes everything after a delimiter from a column in the DataFrame
        """
        if column_name in df.columns:
            return df[column_name].apply(lambda gene_col: None if gene_col is None else gene_col.rsplit(delimiter)[0])

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
                             f"This parameter can either point to a single file with immune receptor data or to a directory containing such files.")

        assert len(
            filenames) >= 1, f"ImportHelper: the dataset {dataset_name} cannot be imported, no files were found under {path}.\n" \
                             f"Note that only files with the following extensions can be imported: {data_file_extensions}"
        return filenames

    @staticmethod
    def import_sequence_dataset(import_class, params, dataset_name: str):
        PathBuilder.build(params.result_path)

        filenames = ImportHelper.get_sequence_filenames(params.path, dataset_name)

        dataset_params = {}
        items = None

        for index, filename in enumerate(filenames):
            new_items = ImportHelper.import_items(import_class, filename, params)
            items = np.append(items, new_items) if items is not None else new_items
            dataset_params = ImportHelper.extract_sequence_dataset_params(items, params)

        cls = ReceptorDataset if params.paired else SequenceDataset
        dataset = cls.build_from_objects(items, params.sequence_file_size, params.result_path, dataset_name,
                                         dataset_params)

        AIRRExporter.export(dataset, params.result_path)

        return dataset

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

    @staticmethod
    def import_items(import_class, path, params: DatasetImportParams):
        alternative_load_func = getattr(import_class, "alternative_load_func", None)
        df = ImportHelper.load_sequence_dataframe(path, params, alternative_load_func)
        df = import_class.preprocess_dataframe(df, params)

        if params.paired:
            import_receptor_func = getattr(import_class, "import_receptors", None)
            if import_receptor_func:
                sequences = import_receptor_func(df, params)
            else:
                raise NotImplementedError(
                    f"{import_class.__name__}: import of paired receptor data has not been implemented.")
        else:
            metadata_columns = params.metadata_column_mapping.values() if params.metadata_column_mapping else None
            sequences = df.apply(ImportHelper.import_sequence, metadata_columns=metadata_columns, axis=1).values

        return sequences

    @classmethod
    def filter_illegal_sequences(cls, df: pd.DataFrame, params: DatasetImportParams, location: str):
        try:

            if not params.import_out_of_frame:
                df = df[df.vj_in_frame]
            if params.import_productive:
                df = df[df.productive]

        except AttributeError as e:
            logging.warning(f"An error occurred while filtering illegal sequences while importing the "
                            f"dataset {location}. Error: {e}\n\nFiltering will be skipped.")

        return df
