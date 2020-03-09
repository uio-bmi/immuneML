import json

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.analysis.data_manipulation.DataReshaper import DataReshaper
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder
from source.visualization.FacetScalesType import FacetScalesType
from source.visualization.FacetSwitchType import FacetSwitchType
from source.visualization.FacetType import FacetType


class DistributionPlot(EncodingReport):
    """
    Generate distribution plots (violin plots, boxplots, density, etc.) of multiple feature, faceted/colored/grouped by
    various example- and feature-level metadata
    @param result_name: string indicating resulting figure file name
    @param result_path: string indicating resulting figure file path
    @param type: type of plot, from DistributionType - can be violin, quasirandom, box, ridge, sina, or line - refer to
    respective ggplot functions for descriptions
    @param x: metadata attribute - either from column names of encoded_dataset.encoded_data.feature_annotations or keys
    of encoded_dataset.encoded_data.labels to use on x-axis of plot
    @param color: metadata attribute to color data by
    @param group: metadata attribute to group points by - only relevant if type is line, it is used to draw lines
    between grouped samples (e.g. for longitudinal data)
    @param palette: dictionary of desired color palette - if discrete, in the form value: color, if continuous, in the
    form colors: list of colors and breaks: list of values at which each color should be - for example:
    discrete: {"A": "blue", "B": "red", ...}
    continuous: {"colors": ["blue", "white", "red"], "breaks": [-1, 0, 1]}
    @param facet_rows: metadata attributes to split points by in rows of the resulting plot-matrix
    @param facet_columns: metadata attributes to split points by in columns of the resulting plot-matrix
    @param facet_type: key defined in FacetType, keep in mind distinction between facet_rows and facet_columns is only relevant
    if grid
    @param facet_scales: key defined in FacetScalesType
    @param facet_switch: key defined in FacetSwitchType
    @param nrow: number of rows to arrange facets in
    @param height: float indicating final figure height
    @param width: float indicating final figure width

    example:

    x="status",
    color="age",
    facet_columns=["week"],
    facet_rows=["feature"],
    facet_type="grid",
    palette={"colors": ["blue", "red"],
    result_path=path,
    result_name="test2",
    height=6
    """

    @classmethod
    def build_object(cls, **kwargs):
        return DistributionPlot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, result_name: str = None,
                 type: str = "quasirandom", x: str = None, color: str = "NULL", group: str = "NULL", palette: dict = {},
                 facet_rows: list = None, facet_columns: list = None, facet_type: str = "grid",
                 facet_scales: str = "free", facet_switch: str = "NULL", nrow: int = 3,
                 height: float = 10, width: float = 10):

        self.dataset = dataset
        self.result_path = result_path
        self.result_name = result_name
        self.type = type
        self.x = x
        self.color = color
        self.group = group
        self.palette = palette
        self.facet_rows = facet_rows if facet_rows is not None else []
        self.facet_columns = facet_columns if facet_columns is not None else []
        self.facet_type = FacetType[facet_type.upper()].name.lower()
        self.facet_scales = FacetScalesType[facet_scales.upper()].name.lower()
        self.facet_switch = FacetSwitchType[facet_switch.upper()].name.lower()
        self.nrow = nrow
        self.height = height
        self.width = width

    def generate(self):
        PathBuilder.build(self.result_path)
        self._plot()

    def _plot(self):

        data = DataReshaper.reshape(self.dataset)

        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/Distributions.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_distributions(data=data, x=self.x, color=self.color, group=self.group, palette=json.dumps(self.palette),
                  facet_rows=self.facet_rows, facet_columns=self.facet_columns, facet_type=self.facet_type,
                  facet_scales=self.facet_scales, facet_switch=self.facet_switch, nrow=self.nrow, height=self.height,
                  width=self.width, result_path=self.result_path, result_name=self.result_name, type=self.type)
