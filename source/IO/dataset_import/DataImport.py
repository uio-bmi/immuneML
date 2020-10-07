# quality: gold

import abc
import pandas as pd
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset


class DataImport(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def import_dataset(params, dataset_name: str) -> Dataset:
        pass

    @staticmethod
    @abc.abstractmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams) -> pd.DataFrame:
        pass

