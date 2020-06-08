import os
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.dataset_import.PickleImport import PickleImport
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class ImportHelper:

    DATASET_FORMAT = "iml_dataset"

    @staticmethod
    def import_or_load_imported(params: dict, processed_params, dataset_name: str, preprocess_repertoire_func):

        dataset_file = f"{processed_params.result_path}{dataset_name}.{ImportHelper.DATASET_FORMAT}"

        if os.path.isfile(dataset_file):
            params["path"] = dataset_file
            dataset = PickleImport.import_dataset(params)
        else:
            dataset = ImportHelper.import_repertoire_dataset(preprocess_repertoire_func, processed_params)

        return dataset

    @staticmethod
    def import_repertoire_dataset(preprocess_repertoire_func, params: DatasetImportParams) -> RepertoireDataset:
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

        PathBuilder.build(params.result_path)

        arguments = [(preprocess_repertoire_func, row, params) for index, row in metadata.iterrows()]
        with Pool(params.batch_size) as pool:
            repertoires = pool.starmap(ImportHelper.load_repertoire, arguments)

        new_metadata_file = ImportHelper.make_new_metadata_file(repertoires, metadata, params.result_path)

        potential_labels = list(set(metadata.columns.tolist()) - {"filename"})
        dataset = RepertoireDataset(params={key: list(set(metadata[key].values.tolist())) for key in potential_labels},
                                    repertoires=repertoires, metadata_file=new_metadata_file)

        PickleExporter.export(dataset, params.result_path)

        return dataset

    @staticmethod
    def make_new_metadata_file(repertoires: list, metadata: pd.DataFrame, result_path: str) -> str:
        new_metadata = metadata.copy()
        new_metadata["filename"] = [os.path.basename(repertoire.data_filename) for repertoire in repertoires]
        new_metadata["identifier"] = [repertoire.identifier for repertoire in repertoires]

        metadata_filename = f"{result_path}metadata.csv"
        new_metadata.to_csv(metadata_filename, index=False, sep=",")

        return metadata_filename

    @staticmethod
    def load_repertoire(preprocess_repertoire_func, metadata, params: DatasetImportParams):

        dataframe = preprocess_repertoire_func(metadata, params)
        sequence_lists = {field: dataframe[field].values.tolist() for field in Repertoire.FIELDS if field in dataframe.columns}
        sequence_lists["custom_lists"] = {field: dataframe[field].values.tolist()
                                          for field in list(set(dataframe.columns) - set(Repertoire.FIELDS))}

        repertoire_inputs = {**{"metadata": metadata.to_dict(), "path": params.result_path}, **sequence_lists}
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

    @staticmethod
    def parse_germline(df: pd.DataFrame, gene_name_replacement: dict, germline_value_replacement: dict):

        if all(item in df.columns for item in ["v_genes", "j_genes"]):

            df[["v_genes", "j_genes"]] = df[["v_genes", "j_genes"]].replace(gene_name_replacement)

        if all(item in df.columns for item in ["v_subgroup", "v_genes", "j_subgroup", "j_genes"]):

            df[["v_subgroup", "v_genes", "j_subgroup", "j_genes"]] = df[
                ["v_subgroup", "v_genes", "j_subgroup", "j_genes"]].replace(germline_value_replacement, regex=True)

            df["v_allele"] = df['v_genes'].str.cat(df['v_allele'], sep=Constants.ALLELE_DELIMITER)
            df["j_allele"] = df['j_genes'].str.cat(df['j_allele'], sep=Constants.ALLELE_DELIMITER)

        return df

    @staticmethod
    def parse_adaptive_germline_to_imgt(dataframe):
        gene_name_replacement = pd.read_csv(
            EnvironmentSettings.root_path + "source/IO/dataset_import/conversion/imgt_adaptive_conversion.csv")
        gene_name_replacement = dict(zip(gene_name_replacement.Adaptive, gene_name_replacement.IMGT))

        germline_value_replacement = {**{"TCRB": "TRB", "TCRA": "TRA"}, **{("0" + str(i)): str(i) for i in range(10)}}

        return ImportHelper.parse_germline(dataframe, gene_name_replacement, germline_value_replacement)
