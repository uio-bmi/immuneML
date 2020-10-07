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
from source.data_model.receptor.BCReceptor import BCReceptor
from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.RegionDefinition import RegionDefinition
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

    # @staticmethod   #t todo deprecated: remove
    # def import_or_load_dataset(params: dict, processed_params, dataset_name: str, preprocess_repertoire_func):
    #
    #     dataset_file = f"{processed_params.result_path}{dataset_name}.{ImportHelper.DATASET_FORMAT}"
    #
    #     if os.path.isfile(dataset_file):
    #         params["path"] = dataset_file
    #         dataset = PickleImport.import_dataset(params, dataset_name)
    #     else:
    #         dataset = ImportHelper.import_repertoire_dataset(preprocess_repertoire_func, processed_params, dataset_name)
    #
    #     return dataset

    @staticmethod
    def import_dataset(import_class, params: dict, dataset_name: str) -> Dataset:
        adaptive_params = DatasetImportParams.build_object(**params)

        dataset = ImportHelper.load_dataset_if_exists(params, adaptive_params, dataset_name)
        if dataset is None:
            if adaptive_params.is_repertoire:
                dataset = ImportHelper.import_repertoire_dataset(import_class.preprocess_repertoire, # todo just pass import_class??? (consistent)
                                                                 adaptive_params, dataset_name)
            else:
                dataset = ImportHelper.import_sequence_dataset(import_class, adaptive_params, dataset_name)

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
    def import_repertoire_dataset(preprocess_repertoire_df_func, params: DatasetImportParams, dataset_name: str) -> RepertoireDataset:
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

        arguments = [(preprocess_repertoire_df_func, row, params) for index, row in metadata.iterrows()]
        with Pool(params.batch_size) as pool:
            repertoires = pool.starmap(ImportHelper.load_repertoire, arguments)

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
    def load_repertoire(preprocess_sequence_df_func, metadata, params: DatasetImportParams):

        dataframe = preprocess_sequence_df_func(metadata, params)
        sequence_lists = {field: dataframe[field].values.tolist() for field in Repertoire.FIELDS if field in dataframe.columns}
        sequence_lists["custom_lists"] = {field: dataframe[field].values.tolist()
                                          for field in list(set(dataframe.columns) - set(Repertoire.FIELDS))}

        repertoire_inputs = {**{"metadata": metadata.to_dict(), "path": params.result_path+"repertoires/"}, **sequence_lists}
        repertoire = Repertoire.build(**repertoire_inputs)

        return repertoire

    @staticmethod
    def load_repertoire_as_dataframe(metadata: dict, params, alternative_load_func=None):
        filepath = f"{params.path}{metadata['filename']}"

        try:
            df = ImportHelper.load_sequence_dataframe(filepath, params, alternative_load_func)
        except Exception as ex:
            raise Exception(f"{ex}\n\nDatasetImport: an error occurred while importing a dataset while parsing the file: {filepath}.\n"
                            f"The parameters used for import are {params}.\nFor technical description of the error, see the log above."
                            f" For details on how to specify the dataset import, see the documentation.")

        return df

    @staticmethod
    def load_sequence_dataframe(filepath, params, alternative_load_func=None):
        if alternative_load_func:
            df = alternative_load_func(filepath, params)
        else:
            df = pd.read_csv(filepath, sep=params.separator, iterator=False, usecols=params.columns_to_load, dtype=str)

        if hasattr(params, "column_mapping") and params.column_mapping is not None:
            df.rename(columns=params.column_mapping, inplace=True)

        df = ImportHelper.standardize_none_values(df)

        return df

    @staticmethod
    def standardize_none_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.replace({key: Constants.UNKNOWN for key in ["unresolved", "no data", "na", "unknown", "null", "nan", np.nan, ""]})

    # @staticmethod # todo old version remove
    # def import_sequence_dataset(sequence_import_func, params, dataset_name: str, *args, **kwargs):
    #     PathBuilder.build(params.result_path)
    #
    #     filenames = [params.path] if os.path.isfile(params.path) else glob(params.path + "*.tsv")
    #     assert len(filenames) >= 1, f"ImportHelper: the dataset {dataset_name} cannot be imported, no files were found under {params.path}."
    #
    #     file_index = 0
    #     dataset_filenames = []
    #     items = None
    #
    #     for index, filename in enumerate(filenames):
    #         new_items = sequence_import_func(filename, *args, **kwargs)
    #         items = np.append(items, new_items) if items is not None else new_items
    #
    #         while len(items) > params.sequence_file_size or (index == len(filenames) - 1 and len(items) > 0):
    #             dataset_filenames.append(params.result_path + "batch_{}.pickle".format(file_index))
    #             ImportHelper.store_sequence_items(dataset_filenames, items, params.sequence_file_size)
    #             items = items[params.sequence_file_size:]
    #             file_index += 1
    #
    #     dataset = ReceptorDataset(filenames=dataset_filenames, file_size=params.sequence_file_size, name=dataset_name) if params.paired \
    #         else SequenceDataset(filenames=dataset_filenames, file_size=params.sequence_file_size, name=dataset_name)
    #
    #     PickleExporter.export(dataset, params.result_path)
    #
    #     return dataset

    @staticmethod #
    def import_sequence_dataset(import_class, params, dataset_name: str): # todo remove args kwargs
        PathBuilder.build(params.result_path)

        filenames = [params.path] if os.path.isfile(params.path) else glob(params.path + "*.tsv")
        assert len(filenames) >= 1, f"ImportHelper: the dataset {dataset_name} cannot be imported, no files were found under {params.path}."

        file_index = 0
        dataset_filenames = []
        items = None

        for index, filename in enumerate(filenames):
            new_items = ImportHelper.import_items(import_class, filename, params)
            # new_items = sequence_import_func(filename, params, *args, **kwargs)
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
        df = ImportHelper.load_sequence_dataframe(path, params)
        df = import_class.preprocess_dataframe(df, params)

        if params.paired:
            sequences = import_class.import_receptors(df, params)
        else:
            sequences = df.apply(ImportHelper.import_sequence, metadata_columns=params.metadata_columns, axis=1).values

        return sequences

    @staticmethod
    def store_sequence_items(dataset_filenames: list, items: list, sequence_file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:sequence_file_size], file)

    @staticmethod
    def parse_adaptive_germline_to_imgt(dataframe):
        gene_name_replacement = pd.read_csv(
            EnvironmentSettings.root_path + "source/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        gene_name_replacement = dict(zip(gene_name_replacement.Adaptive, gene_name_replacement.IMGT))

        germline_value_replacement = {**{"TCRB": "TRB", "TCRA": "TRA"}, **{("0" + str(i)): str(i) for i in range(10)}}

        return ImportHelper.parse_germline(dataframe, gene_name_replacement, germline_value_replacement)

    @staticmethod
    def parse_germline(df: pd.DataFrame, gene_name_replacement: dict, germline_value_replacement: dict):

        if all(item in df.columns for item in ["v_genes", "j_genes"]):

            df[["v_genes", "j_genes"]] = df[["v_genes", "j_genes"]].replace(gene_name_replacement)

        if all(item in df.columns for item in ["v_subgroups", "v_genes", "j_subgroups", "j_genes"]):

            df[["v_subgroups", "v_genes", "j_subgroups", "j_genes"]] = df[
                ["v_subgroups", "v_genes", "j_subgroups", "j_genes"]].replace(germline_value_replacement, regex=True)

        if all(item in df.columns for item in ["v_genes", "j_genes", "v_alleles", "j_alleles"]):

            df["v_alleles"] = df['v_genes'].str.cat(df['v_alleles'], sep=Constants.ALLELE_DELIMITER)
            df["j_alleles"] = df['j_genes'].str.cat(df['j_alleles'], sep=Constants.ALLELE_DELIMITER)

        return df

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
    def junction_to_cdr3(df: pd.DataFrame, region_definition: RegionDefinition, region_type: RegionType):
        '''
        If RegionType is CDR3, the leading C and trailing W are removed from the sequence to match the CDR3 definition.
        This method alters the data in the provided dataframe.

        See :py:obj:`~source.data_model.receptor.RegionDefinition.RegionDefinition` description.
        '''

        if region_definition == RegionDefinition.IMGT and region_type == RegionType.CDR3:
            df["sequence_aas"] = df["sequence_aas"].str[1:-1]
            df["sequences"] = df["sequences"].str[3:-3]
            df["region_types"] = region_type.name

    @staticmethod
    def strip_alleles(df: pd.DataFrame, column_name):
        '''
        Removes alleles (everythin after the '*' character) from a column in the DataFrame
        '''
        return df[column_name].apply(lambda gene_col: gene_col.rsplit("*", maxsplit=1)[0])

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
            receptors = ImportHelper.import_receptors_for_id(df, identifier, params)
            all_receptors.extend(receptors)

        return all_receptors


    @staticmethod
    def import_receptors_for_id(df, identifier, params) -> List[Receptor]:
        first_row = df.loc[(df["receptor_identifiers"] == identifier) & (df["chains"] == params.receptor_chains.value[0])]
        second_row = df.loc[(df["receptor_identifiers"] == identifier) & (df["chains"] == params.receptor_chains.value[1])]

        for i, row in enumerate([first_row, second_row]):
            if row.shape[0] > 1:
                warnings.warn(f"Multiple {params.receptor_chains.value[i]} chains found for receptor with identifier {identifier}, only the first entry will be loaded")
            elif row.shape[0] == 0:
                warnings.warn(f"Missing {params.receptor_chains.value[i]} chain for receptor with identifier {identifier}, this receptor will be omitted.")
                return []

        # todo: add 'import all' functionality like IRIS import, to handle dual chains (all receptors / all chains / just one / different chain combos??)

        return [ImportHelper.build_receptor_from_rows(first_row.iloc[0], second_row.iloc[0], identifier, params)]


    @staticmethod
    def build_receptor_from_rows(first_row, second_row, identifier, params):
        first_sequence = ImportHelper.import_sequence(first_row, metadata_columns=params.metadata_columns)
        second_sequence = ImportHelper.import_sequence(second_row, metadata_columns=params.metadata_columns)

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
        elif params.receptor_chains == ChainPair.IGH_IGK:
            receptor = BCReceptor(heavy=first_sequence,
                                  light=second_sequence,
                                  identifier=identifier,
                                  metadata={**first_sequence.metadata.custom_params})
        else:
            raise NotImplementedError(f"ImportHelper: {params.receptor_chains} chain pair is not supported.")

        return receptor

