import os
import pickle
import warnings
from glob import glob
from multiprocessing.pool import Pool

from typing import List
import numpy as np
import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.dataset_import.PickleImport import PickleImport
from source.data_model.dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.BCKReceptor import BCKReceptor
from source.data_model.receptor.BCReceptor import BCReceptor
from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.RegionType import RegionType
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.TCGDReceptor import TCGDReceptor
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class ImportHelper:

    DATASET_FORMAT = "iml_dataset"

    @staticmethod
    def import_dataset(import_class, params: dict, dataset_name: str) -> Dataset:
        processed_params = DatasetImportParams.build_object(**params)

        dataset = ImportHelper.load_dataset_if_exists(params, processed_params, dataset_name)
        if dataset is None:
            # backwards compatability: if is_repertoire is not specified but the metadata file is
            if processed_params.is_repertoire is None and processed_params.metadata_file is not None:
                processed_params.is_repertoire = True

            if processed_params.is_repertoire:
                dataset = ImportHelper.import_repertoire_dataset(import_class, processed_params, dataset_name)
            else:
                dataset = ImportHelper.import_sequence_dataset(import_class, processed_params, dataset_name)

        return dataset

    @staticmethod
    def load_dataset_if_exists(params: dict, processed_params, dataset_name: str):

        dataset_file = f"{processed_params.result_path}{dataset_name}.{ImportHelper.DATASET_FORMAT}"
        dataset = None

        if os.path.isfile(dataset_file):
            params["path"] = dataset_file
            dataset = PickleImport.import_dataset(params, dataset_name)

        return dataset


    @staticmethod
    def import_repertoire_dataset(import_class, params: DatasetImportParams, dataset_name: str) -> RepertoireDataset:
        """
        Function to create a dataset from the metadata and a list of repertoire files and exports dataset pickle file

        Arguments:
            preprocess_repertoire_df_func: function which should preprocess a dataframe of sequences given:
                the metadata row for the repertoire,
                params object of DatasetImportParams class with path, result path, and other info for import
            params: instance of DatasetImportParams class which includes information on path, columns, result path etc.
        Returns:
            RepertoireDataset object that was created
        """
        metadata = pd.read_csv(params.metadata_file, ",")

        PathBuilder.build(params.result_path+"repertoires/")

        arguments = [(import_class, row, params) for index, row in metadata.iterrows()]
        with Pool(params.batch_size) as pool:
            repertoires = pool.starmap(ImportHelper.load_repertoire_as_object, arguments)

        new_metadata_file = ImportHelper.make_new_metadata_file(repertoires, metadata, params.result_path, dataset_name)

        potential_labels = list(set(metadata.columns.tolist()) - {"filename"})
        dataset = RepertoireDataset(params={key: list(set(metadata[key].values.tolist())) for key in potential_labels},
                                    repertoires=repertoires, metadata_file=new_metadata_file, name=dataset_name)

        PickleExporter.export(dataset, params.result_path)

        return dataset

    @staticmethod
    def make_new_metadata_file(repertoires: list, metadata: pd.DataFrame, result_path: str, dataset_name: str) -> str:
        new_metadata = metadata.copy()
        new_metadata["filename"] = [os.path.basename(repertoire.data_filename) for repertoire in repertoires]
        new_metadata["identifier"] = [repertoire.identifier for repertoire in repertoires]

        metadata_filename = f"{result_path}{dataset_name}_metadata.csv"
        new_metadata.to_csv(metadata_filename, index=False, sep=",")

        return metadata_filename

    @staticmethod
    def load_repertoire_as_object(import_class, metadata_row, params: DatasetImportParams):
        alternative_load_func = getattr(import_class, "alternative_load_func", None)

        dataframe = ImportHelper.load_sequence_dataframe(f"{params.path}{metadata_row['filename']}", params, alternative_load_func)
        dataframe = import_class.preprocess_dataframe(dataframe, params)
        sequence_lists = {field: dataframe[field].values.tolist() for field in Repertoire.FIELDS if field in dataframe.columns}
        sequence_lists["custom_lists"] = {field: dataframe[field].values.tolist()
                                          for field in list(set(dataframe.columns) - set(Repertoire.FIELDS))}

        repertoire_inputs = {**{"metadata": metadata_row.to_dict(), "path": params.result_path+"repertoires/"}, **sequence_lists}
        repertoire = Repertoire.build(**repertoire_inputs)

        return repertoire

    @staticmethod
    def load_sequence_dataframe(filepath, params, alternative_load_func=None):
        try:
            if alternative_load_func:
                df = alternative_load_func(filepath, params)
            else:
                df = pd.read_csv(filepath, sep=params.separator, iterator=False, usecols=params.columns_to_load, dtype=str)
        except Exception as ex:
            raise Exception(f"{ex}\n\nImportHelper: an error occurred during dataset import while parsing the input file: {filepath}.\n"
                            f"Please make sure this is a correct immune receptor data file (not metadata).\n"
                            f"The parameters used for import are {params}.\nFor technical description of the error, see the log above."
                            f" For details on how to specify the dataset import, see the documentation.")

        if hasattr(params, "column_mapping") and params.column_mapping is not None:
            df.rename(columns=params.column_mapping, inplace=True)

        if hasattr(params, "metadata_column_mapping") and params.metadata_column_mapping is not None:
            df.rename(columns=params.metadata_column_mapping, inplace=True)

        df = ImportHelper.standardize_none_values(df)

        return df

    @staticmethod
    def standardize_none_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.replace({key: Constants.UNKNOWN for key in ["unresolved", "no data", "na", "unknown", "null", "nan", np.nan, ""]})

    @staticmethod
    def prepare_frame_type_list(params: DatasetImportParams) -> list:
        frame_type_list = []
        if params.import_productive:
            frame_type_list.append(SequenceFrameType.IN.name)
        if params.import_out_of_frame:
            frame_type_list.append(SequenceFrameType.OUT.name)
        if params.import_with_stop_codon:
            frame_type_list.append(SequenceFrameType.STOP.name)
        return frame_type_list

    @staticmethod
    def load_chains_from_genes(df: pd.DataFrame, column_name) -> list:
        return [Chain.get_chain(chain_str) for chain_str in df[column_name].str[0:3]]

    @staticmethod
    def junction_to_cdr3(df: pd.DataFrame, region_type: RegionType):
        '''
        If RegionType is CDR3, the leading C and trailing W are removed from the sequence to match the IMGT CDR3 definition.
        This method alters the data in the provided dataframe.
        '''

        if region_type == RegionType.IMGT_CDR3:
            if "sequence_aas" in df:
                df["sequence_aas"] = df["sequence_aas"].str[1:-1]
            if "sequences" in df:
                df["sequences"] = df["sequences"].str[3:-3]
            df["region_types"] = region_type.name

    @staticmethod
    def strip_alleles(df: pd.DataFrame, column_name):
        '''
        Removes alleles (everythin after the '*' character) from a column in the DataFrame
        '''
        return df[column_name].apply(lambda gene_col: gene_col.rsplit("*", maxsplit=1)[0])

    @staticmethod
    def get_sequence_filenames(path, dataset_name):
        data_file_extensions = ("*.tsv", "*.csv", "*.txt")

        if os.path.isfile(path):
            filenames = [path]
        elif os.path.isdir(path):
            filenames = []

            for pattern in data_file_extensions:
                filenames.extend(glob(os.path.join(path, pattern)))
        else:
            raise ValueError(f"ImportHelper: path '{path}' given in YAML specification is not a valid path. "
                             f"This parameter can either point to a single file with immune receptor data or to a directory containing such files.")

        assert len(filenames) >= 1, f"ImportHelper: the dataset {dataset_name} cannot be imported, no files were found under {path}.\n" \
                                    f"Note that only files with the following extensions can be imported: {data_file_extensions}"
        return filenames

    @staticmethod
    def import_sequence_dataset(import_class, params, dataset_name: str):
        PathBuilder.build(params.result_path)

        filenames = ImportHelper.get_sequence_filenames(params.path, dataset_name)

        file_index = 0
        dataset_filenames = []
        items = None

        for index, filename in enumerate(filenames):
            new_items = ImportHelper.import_items(import_class, filename, params)
            items = np.append(items, new_items) if items is not None else new_items

            while len(items) > params.sequence_file_size or (index == len(filenames) - 1 and len(items) > 0):
                dataset_filenames.append(params.result_path + "batch_{}.pickle".format(file_index))
                ImportHelper.store_sequence_items(dataset_filenames, items, params.sequence_file_size)
                items = items[params.sequence_file_size:]
                file_index += 1

        dataset = ReceptorDataset(filenames=dataset_filenames, file_size=params.sequence_file_size, name=dataset_name) if params.paired \
            else SequenceDataset(filenames=dataset_filenames, file_size=params.sequence_file_size, name=dataset_name)

        PickleExporter.export(dataset, params.result_path)

        return dataset

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
                raise NotImplementedError(f"{import_class.__name__}: import of paired receptor data has not been implemented.")
        else:
            metadata_columns = params.metadata_column_mapping.values() if params.metadata_column_mapping else None
            sequences = df.apply(ImportHelper.import_sequence, metadata_columns=metadata_columns, axis=1).values

        return sequences

    @staticmethod
    def store_sequence_items(dataset_filenames: list, items: list, sequence_file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:sequence_file_size], file)

    @staticmethod
    def import_sequence(row, metadata_columns=[]) -> ReceptorSequence:
        metadata = SequenceMetadata(v_gene=str(row["v_genes"]) if "v_genes" in row and row["v_genes"] is not None else None,
                                    j_gene=str(row["j_genes"]) if "j_genes" in row and row["j_genes"] is not None else None,
                                    chain=row["chains"] if "chains" in row and row["chains"] is not None else None,
                                    region_type=row["region_types"] if "region_types" in row and row["region_types"] is not None else None,
                                    count=int(row["counts"]) if "counts" in row and row["counts"] is not None else None,
                                    frame_type=row["frame_types"] if "frame_types" in row and row["frame_types"] is not None else None,
                                    custom_params={custom_col: row[custom_col] for custom_col in metadata_columns if custom_col in row} if metadata_columns is not None else {})
        sequence = ReceptorSequence(amino_acid_sequence=str(row["sequence_aas"]) if "sequence_aas" in row and row["sequence_aas"] is not None else None,
                                    nucleotide_sequence=str(row["sequences"]) if "sequences" in row and row["sequences"] is not None else None,
                                    identifier=str(row["sequence_identifiers"]) if "sequence_identifiers" in row and row["sequence_identifiers"] is not None else None,
                                    metadata=metadata)

        return sequence

    @staticmethod
    def import_receptors(df, params) -> List[Receptor]:
        identifiers = df["receptor_identifiers"].unique()
        all_receptors = []

        for identifier in identifiers:
            receptors = ImportHelper.import_receptors_by_id(df, identifier, params)
            all_receptors.extend(receptors)

        return all_receptors

    @staticmethod
    def import_receptors_by_id(df, identifier, params) -> List[Receptor]:
        first_row = df.loc[(df["receptor_identifiers"] == identifier) & (df["chains"] == params.receptor_chains.value[0])]
        second_row = df.loc[(df["receptor_identifiers"] == identifier) & (df["chains"] == params.receptor_chains.value[1])]

        for i, row in enumerate([first_row, second_row]):
            if row.shape[0] > 1:
                warnings.warn(f"Multiple {params.receptor_chains.value[i]} chains found for receptor with identifier {identifier}, only the first entry will be loaded")
            elif row.shape[0] == 0:
                warnings.warn(f"Missing {params.receptor_chains.value[i]} chain for receptor with identifier {identifier}, this receptor will be omitted.")
                return []

        # todo add options like IRIS import: option to import all dual chains or just the first pair / all V genes when uncertain annotation, etc
        # todo add possibility to import multiple chain combo's? (BCR heavy-light & heavy-kappa, as seen in 10xGenomics?)

        return [ImportHelper.build_receptor_from_rows(first_row.iloc[0], second_row.iloc[0], identifier, params)]

    @staticmethod
    def build_receptor_from_rows(first_row, second_row, identifier, params):
        metadata_columns = params.metadata_column_mapping.values() if params.metadata_column_mapping else None
        first_sequence = ImportHelper.import_sequence(first_row, metadata_columns=metadata_columns)
        second_sequence = ImportHelper.import_sequence(second_row, metadata_columns=metadata_columns)

        if params.receptor_chains == ChainPair.TRA_TRB:
            receptor = TCABReceptor(alpha=first_sequence,
                                    beta=second_sequence,
                                    identifier=identifier,
                                    metadata={**second_sequence.metadata.custom_params})
        elif params.receptor_chains == ChainPair.TRG_TRD:
            receptor = TCGDReceptor(gamma=first_sequence,
                                    delta=second_sequence,
                                    identifier=identifier,
                                    metadata={**second_sequence.metadata.custom_params})
        elif params.receptor_chains == ChainPair.IGH_IGL:
            receptor = BCReceptor(heavy=first_sequence,
                                  light=second_sequence,
                                  identifier=identifier,
                                  metadata={**first_sequence.metadata.custom_params})
        elif params.receptor_chains == ChainPair.IGH_IGK:
            receptor = BCKReceptor(heavy=first_sequence,
                                  kappa=second_sequence,
                                  identifier=identifier,
                                  metadata={**first_sequence.metadata.custom_params})
        else:
            raise NotImplementedError(f"ImportHelper: {params.receptor_chains} chain pair is not supported.")

        return receptor

