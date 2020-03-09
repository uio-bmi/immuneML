import copy
import json

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.analysis.similarities.RepertoireSimilarityComputer import RepertoireSimilarityComputer
from source.analysis.similarities.SimilarityMeasureType import SimilarityMeasureType

pandas2ri.activate()

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class SimilarityHeatmap(EncodingReport):

    """
    Plot pairwise similarity between repertoires in a heatmap.
    @param similarity_measure: value from SimilarityMeasureType indicating similarity measure to be computed and plotted
    Refer to documentation of FeatureHeatmap, for overlapping parameters, the definitions are identical.

    example:

    similarity_measure="PEARSON",
    one_hot_encode_example_annotations=["disease"],
    example_annotations=["age", "week"],
    palette={"week": {"4": ["white"]}, "age": {"colors": ["cornflowerblue", "white", "firebrick"], "breaks": [40, 50, 60]}},
    annotation_position="left",
    show_names=True,
    names_size=0.5,
    height=5,
    width=6.7,
    text_size=6,
    result_path=path
    """

    FEATURE = "feature"
    EXAMPLE = "example"

    @classmethod
    def build_object(cls, **kwargs):
        return SimilarityHeatmap(**kwargs)

    def __init__(self,
                 dataset: RepertoireDataset = None,
                 similarity_measure: str = "jaccard",
                 example_annotations: list = [],
                 one_hot_encode_example_annotations: list = [],
                 palette: dict = {},
                 cluster: bool = True,
                 show_dend: bool = True,
                 show_names: bool = False,
                 show_legend: list = None,
                 annotation_position: str = "top",
                 legend_position: str = "side",
                 text_size: float = 10,
                 names_size: float = 7,
                 height: float = 10,
                 width: float = 10,
                 result_name: str = "similarity_heatmap",
                 result_path: str = None):

        self.dataset = dataset
        self.similarity_measure = SimilarityMeasureType[similarity_measure.upper()]
        self.annotations = example_annotations
        self.one_hot_encode_annotations = one_hot_encode_example_annotations
        self.palette = palette
        self.cluster = cluster
        self.show_dend = show_dend
        self.show_names = show_names
        self.show_legend = show_legend
        self.annotation_position = annotation_position
        self.legend_position = legend_position
        self.text_size = text_size
        self.names_size = names_size
        self.height = height
        self.width = width
        self.result_name = result_name
        self.result_path = result_path

        if self.show_legend is None:
            self.show_legend = copy.deepcopy(self.annotations)

    def generate(self):
        PathBuilder.build(self.result_path)
        self._plot()

    def _plot(self):

        matrix = self._compute_similarity_matrix()
        example_annotations = self._prepare_annotations(pd.DataFrame(self.dataset.encoded_data.labels))
        if self.show_legend is None:
            self.show_legend = self.annotations

        with open(EnvironmentSettings.root_path + "source/visualization/Heatmap.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        if self.annotation_position == "top":
            row_annotations = []
            column_annotations = example_annotations
            show_row_dend = False
            show_column_dend = self.show_dend
            show_row_names = False
            show_column_names = self.show_names
            show_legend_row = []
            show_legend_column = self.show_legend
        else:
            row_annotations = example_annotations
            column_annotations = []
            show_row_dend = self.show_dend
            show_column_dend = False
            show_row_names = self.show_names
            show_column_names = False
            show_legend_row = self.show_legend
            show_legend_column = False

        plot.plot_heatmap(matrix=matrix,
                          row_annotations=row_annotations,
                          column_annotations=column_annotations,
                          palette=json.dumps(self.palette),
                          row_names=self.dataset.encoded_data.example_ids,
                          column_names=self.dataset.encoded_data.example_ids,
                          cluster_rows=self.cluster,
                          cluster_columns=self.cluster,
                          show_row_dend=show_row_dend,
                          show_column_dend=show_column_dend,
                          show_row_names=show_row_names,
                          show_column_names=show_column_names,
                          show_legend_row=show_legend_row,
                          show_legend_column=show_legend_column,
                          legend_position=self.legend_position,
                          text_size=self.text_size,
                          row_names_size=self.names_size,
                          column_names_size=self.names_size,
                          scale_rows=False,
                          height=self.height,
                          width=self.width,
                          result_path=self.result_path,
                          result_name=self.result_name)

    def _prepare_annotations(self, data):
        for col in self.one_hot_encode_annotations:
            data = self._one_hot_encode_column(data, col)
        return data[self.annotations]

    def _one_hot_encode_column(self, data, column_name):
        one_hot = pd.get_dummies(data[column_name], dtype=np.float64)
        data = data.drop(column_name, axis=1)
        data = data.join(one_hot)
        self.annotations.extend(one_hot.columns)
        return data

    def _compute_similarity_matrix(self):
        a = np.nan_to_num(getattr(RepertoireSimilarityComputer, "compute_" + self.similarity_measure.name.lower())(self.dataset.encoded_data.examples).A)
        np.fill_diagonal(a, np.nan)
        return a