import copy
import json

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

pandas2ri.activate()

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class DensityHeatmap(EncodingReport):

    """
    Plot density distribution of each feature in encoded data matrix.
    Refer to documentation of FeatureHeatmap, for overlapping parameters, the definitions are identical.

    example:

    scale_features=False,
    feature_annotations=["antigen"],
    palette={"week": {"0": "#BE9764"}, "antigen": {"GAD": "cornflowerblue", "INSB": "firebrick"},
             "age": {"colors": ["blue", "white", "red"], "breaks": [0, 20, 100]}},
    result_path=path,
    show_feature_names=True,
    feature_names_size=7,
    text_size=9,
    height=6,
    width=6
    """

    FEATURE = "feature"

    @classmethod
    def build_object(cls, **kwargs):
        return DensityHeatmap(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, feature_annotations: list = [], one_hot_encode_feature_annotations: list = [],
                 palette: dict = {}, cluster_features: bool = True, subset_nonzero_features: bool = False, show_feature_dend: bool = True,
                 show_feature_names: bool = False, show_legend_features: list = None, legend_position: str = "side", text_size: float = 10,
                 feature_names_size: float = 7, scale_features: bool = True, height: float = 10, width: float = 10,
                 result_name: str = "feature_heatmap", result_path: str = None):

        super().__init__()
        self.dataset = dataset
        self.feature_annotations = list(set(feature_annotations) - set(one_hot_encode_feature_annotations))
        self.one_hot_encode_feature_annotations = one_hot_encode_feature_annotations
        self.palette = palette
        self.cluster_features = cluster_features
        self.subset_nonzero_features = subset_nonzero_features
        self.show_feature_dend = show_feature_dend
        self.show_feature_names = show_feature_names
        self.show_legend_features = show_legend_features
        self.legend_position = legend_position
        self.text_size = text_size
        self.feature_names_size = feature_names_size
        self.scale_features = scale_features
        self.height = height
        self.width = width
        self.result_name = result_name
        self.result_path = result_path

        if self.show_legend_features is None:
            self.show_legend_features = copy.deepcopy(self.feature_annotations)

    def generate(self):
        PathBuilder.build(self.result_path)
        self._safe_plot(output_written=False)

    def _plot(self):

        matrix = self._prepare_matrix()
        feature_annotations = self._prepare_annotations(self.dataset.encoded_data.feature_annotations,
                                                        DensityHeatmap.FEATURE)

        with open(EnvironmentSettings.root_path + "source/visualization/DensityHeatmap.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_density_heatmap(matrix=matrix,
                                  feature_annotations=feature_annotations,
                                  palette=json.dumps(self.palette),
                                  feature_names=self.dataset.encoded_data.feature_names,
                                  cluster_features=self.cluster_features,
                                  show_feature_dend=self.show_feature_dend,
                                  show_feature_names=self.show_feature_names,
                                  show_legend_features=self.show_legend_features,
                                  legend_position=self.legend_position,
                                  text_size=self.text_size,
                                  feature_names_size=self.feature_names_size,
                                  scale_features=self.scale_features,
                                  height=self.height,
                                  width=self.width,
                                  result_path=self.result_path,
                                  result_name=self.result_name)

    def _prepare_matrix(self):
        matrix = self.dataset.encoded_data.examples.A.T
        if self.subset_nonzero_features:
            nonzero = np.sum(matrix, axis=1) > 0
            matrix = matrix[nonzero]
            self.dataset.encoded_data.feature_annotations = self.dataset.encoded_data.feature_annotations[nonzero]
            self.dataset.encoded_data.feature_names = [self.dataset.encoded_data.feature_names[i] for i in
                                                       range(nonzero.shape[0]) if nonzero[i]]
        return matrix.T

    def _prepare_annotations(self, data, type):
        for col in getattr(self, "one_hot_encode_" + type + "_annotations"):
            data = self._one_hot_encode_column(data, col, type)
        return data[getattr(self, type + "_annotations")]

    def _one_hot_encode_column(self, data, column_name, type):
        one_hot = pd.get_dummies(data[column_name], dtype=np.float64)
        data = data.drop(column_name, axis=1)
        data = data.join(one_hot)
        getattr(self, type + "_annotations").extend(one_hot.columns)
        return data
