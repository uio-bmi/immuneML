from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class DataFrameWrapper:

    def __init__(self, path: Path, df: pd.DataFrame = None):
        self.path = path
        self.df = df

        if df is not None and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(path), index=False)

    def get_df(self):
        if self.df is None and self.path.exists():
            self.df = pd.read_csv(str(self.path))
        return self.df


@dataclass
class ClusteringSetting:
    encoder: DatasetEncoder
    encoder_params: dict
    encoder_name: str
    clustering_method: ClusteringMethod
    clustering_params: dict
    clustering_method_name: str
    dim_reduction_method: DimRedMethod = None
    dim_red_params: dict = None
    dim_red_name: str = None
    path: Path = None

    def get_key(self) -> str:
        key = self.encoder_name
        if self.dim_red_name:
            key += f"_{self.dim_red_name}"
        key += f"_{self.clustering_method_name}"
        return key

    def __str__(self):
        return self.get_key()


@dataclass
class ClusteringItem:
    dataset: Dataset = None
    method: ClusteringMethod = None
    encoder: DatasetEncoder = None
    internal_performance: DataFrameWrapper = None
    external_performance: DataFrameWrapper = None
    predictions: np.ndarray = None
    cl_setting: ClusteringSetting = None
