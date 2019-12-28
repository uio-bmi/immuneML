import json

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.analysis.data_manipulation.DataReshaper import DataReshaper
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class Barplot(EncodingReport):

    """
    Plot barplot with standard error for features in encoded data matrix.
    Refer to documentation of Distributions, for overlapping parameters, the definitions are identical.

    example:

    result_path=path,
    result_name="test",
    x="status",
    color="age",
    facet_columns=["week"],
    facet_rows=["feature"],
    height=6,
    palette={"NR": "yellow"}
    """

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, result_name: str = None,
                 x: str = None, color: str = "NULL", palette: dict = {},
                 facet_rows: list = None, facet_columns: list = None, facet_type: str = "grid",
                 facet_scales: str = "free", facet_switch: str = "NULL", nrow: int = 3,
                 height: float = 10, width: float = 10):
        self.dataset = dataset
        self.result_path = result_path
        self.result_name = result_name
        self.x = x
        self.color = color
        self.palette = palette
        self.facet_rows = facet_rows if facet_rows is not None else []
        self.facet_columns = facet_columns if facet_columns is not None else []
        self.facet_type = facet_type
        self.facet_scales = facet_scales
        self.facet_switch = facet_switch
        self.nrow = nrow
        self.height = height
        self.width = width

    def generate(self):
        PathBuilder.build(self.result_path)
        self._plot()

    def _plot(self):

        data = DataReshaper.reshape(self.dataset)

        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/Barplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_barplot(data=data, x=self.x, color=self.color, palette=json.dumps(self.palette),
                  facet_rows=self.facet_rows, facet_columns=self.facet_columns, facet_type=self.facet_type,
                  facet_scales=self.facet_scales, facet_switch=self.facet_switch, nrow=self.nrow, height=self.height,
                  width=self.width, result_path=self.result_path, result_name=self.result_name)
