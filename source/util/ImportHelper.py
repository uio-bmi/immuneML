import os
import pickle
from glob import glob
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.dataset_import.PickleImport import PickleImport
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
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
    def load_dataset_if_exists(params: dict, processed_params, dataset_name: str):

        dataset_file = f"{processed_params.result_path}{dataset_name}.{ImportHelper.DATASET_FORMAT}"
        dataset = None

        if os.path.isfile(dataset_file):
            params["path"] = dataset_file
            dataset = PickleImport.import_dataset(params, dataset_name)

        return dataset


    @staticmethod
    def import_repertoire_dataset(preprocess_repertoire_func, params: DatasetImportParams, dataset_name: str) -> RepertoireDataset:
        """
        Function to create a dataset from the metadata and a list of repertoire files and exports dataset pickle file

        Arguments:
            preprocess_repertoire_func: function which should preprocess a repertoire given:
                the metadata row for the repertoire,
                params object of DatasetImportParams class with path, result path, and other info for import
            params: instance of DatasetImportParams class which includes information on path, columns, result path etc.
        Returns:
            RepertoireDataset object that was created
        """
        metadata = pd.read_csv(params.metadata_file, ",")

        PathBuilder.build(params.result_path+"repertoires/")

        arguments = [(preprocess_repertoire_func, row, params) for index, row in metadata.iterrows()]
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
    def load_repertoire(preprocess_repertoire_func, metadata, params: DatasetImportParams):

        dataframe = preprocess_repertoire_func(metadata, params)
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
            if alternative_load_func:
                df = alternative_load_func(filepath, params)
            else:
                df = pd.read_csv(filepath, sep=params.separator, iterator=False, usecols=params.columns_to_load, dtype=str)

            if hasattr(params, "column_mapping") and params.column_mapping is not None:
                df.rename(columns=params.column_mapping, inplace=True)

            df = ImportHelper.standardize_none_values(df)
        except Exception as ex:
            raise Exception(f"{ex}\n\nDatasetImport: an error occurred while importing a dataset while parsing the file: {filepath}.\n"
                            f"The parameters used for import are {params}.\nFor technical description of the error, see the log above."
                            f" For details on how to specify the dataset import, see the documentation.")

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
    def import_sequence_dataset(sequence_import_func, params, dataset_name: str):
        PathBuilder.build(params.result_path)

        filenames = [params.path] if os.path.isfile(params.path) else glob(params.path + "*.tsv")
        assert len(filenames) >= 1, f"ImportHelper: the dataset {dataset_name} cannot be imported, no files were found under {params.path}."

        file_index = 0
        dataset_filenames = []
        items = None

        for index, filename in enumerate(filenames):
            new_items = sequence_import_func(filename, params)
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
    def store_sequence_items(dataset_filenames: list, items: list, sequence_file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:sequence_file_size], file)

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
    def parse_adaptive_germline_to_imgt(dataframe):
        gene_name_replacement = pd.read_csv(
            EnvironmentSettings.root_path + "source/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        gene_name_replacement = dict(zip(gene_name_replacement.Adaptive, gene_name_replacement.IMGT))

        germline_value_replacement = {**{"TCRB": "TRB", "TCRA": "TRA"}, **{("0" + str(i)): str(i) for i in range(10)}}

        return ImportHelper.parse_germline(dataframe, gene_name_replacement, germline_value_replacement)

    @staticmethod
    def prepare_frame_type_list(params: DatasetImportParams) -> list:
        frame_type_list = []
        if params.import_productive:
            frame_type_list.append("In")
        if params.import_out_of_frame:
            frame_type_list.append("Out")
        if params.import_with_stop_codon:
            frame_type_list.append("Stop")
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
    def import_sequence(row):
        if "stop_codon" in row and row["stop_codon"]:
            frame_type = SequenceFrameType.STOP.name
        elif row["productive"]:
            frame_type = SequenceFrameType.IN.name
        elif "vj_in_frame" in row and row["vj_in_frame"]:
            frame_type = SequenceFrameType.IN.name
        else:
            frame_type = SequenceFrameType.OUT.name

        metadata = SequenceMetadata(v_gene=str(row["v_genes"]) if "v_genes" in row else None,
                                    j_gene=str(row["j_genes"]) if "j_genes" in row else None,
                                    chain=row["chains"] if "chains" in row else None,
                                    region_type=row["region_type"] if "region_type" in row else None,
                                    count=int(row["counts"]) if "counts" in row else None,
                                    frame_type=frame_type,
                                    custom_params={"rev_comp": row["rev_comp"]} if "rev_comp" in row else {})
        sequence = ReceptorSequence(amino_acid_sequence=str(row["sequence_aas"]) if "sequence_aas" in row else None,
                                    nucleotide_sequence=str(row["sequences"]) if "sequences" in row else None,
                                    identifier=str(row["sequence_identifiers"]) if "sequence_identifiers" in row else None,
                                    metadata=metadata)

        # todo custom params? epitope etcetera??? --> see VDJdbImport

        return sequence
