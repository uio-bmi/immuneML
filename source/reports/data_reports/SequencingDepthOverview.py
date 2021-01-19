import json

import pandas as pd
from pathlib import Path

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.data_reports.DataReport import DataReport
from source.util.PathBuilder import PathBuilder


class SequencingDepthOverview(DataReport):
    """
    This report plots the sequencing depth as quantified by total number of reads per sample and total number of unique
    clonotypes per sample for in-frame, out-of-frame, and sequences containing stop codons.
    The distributions of each of these variables as well as the relationship between the variables are plotted in this
    report.


    Arguments:

        x (str): discrete column name from metadata file with which to put on the x-axis and split samples by

        color (str): column name from metadata file to color samples by. If no color is specified, x is used.

        facets (list): metadata attributes to split points by in rows of the resulting plot-matrix

        palette (dict): list of colors and breaks: list of values at which each color should be - for example:
        discrete: {"A": "blue", "B": "red", ...}
        continuous: {"colors": ["blue", "white", "red"], "breaks": [-1, 0, 1]}

        nrow_distributions (int): The number of rows used for the distribution plot facets.

        nrow_scatterplot (int): The number of rows used for the scatterplot facets.

        height_distributions (float): Height (in inches) of the distribution section of the resulting plot

        height_scatterplot (float): Height (in inches) of the scatterplot section of the resulting plot

        width (float): Width (in inches) of resulting plot


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_seqdepth_report:
            SequencingDepthOverview:
                x: disease
                palette:
                    disease_1: red
                    disease_2: green
    """

    @classmethod
    def build_object(cls, **kwargs):
        return SequencingDepthOverview(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None,
                 x: str = None,
                 color: str = None,
                 facets=None,
                 palette: dict = None,
                 nrow_distributions: int = 2,
                 nrow_scatterplot: int = 1,
                 height_distributions: float = 6.67,
                 height_scatterplot: float = 3.33,
                 width: float = 10,
                 result_name: str = "sequencing_depth_overview",
                 result_path: Path = None,
                 number_of_processes: int = 1,
                 name: str = None):

        DataReport.__init__(self, dataset=dataset, result_path=result_path, name=name)

        self.x = x
        self.color = color if color is not None else x
        self.facets = facets if facets is not None else []
        self.palette = palette if palette is not None else {}
        self.nrow_distributions = nrow_distributions
        self.nrow_scatterplot = nrow_scatterplot
        self.height_distributions = height_distributions
        self.height_scatterplot = height_scatterplot
        self.width = width
        self.result_name = result_name
        self.number_of_processes = number_of_processes

    def _generate(self) -> ReportResult:
        data = self._generate_data()
        report_output_fig = self._safe_plot(data=data, output_written=False)
        output_figures = [report_output_fig] if report_output_fig is not None else []

        return ReportResult(self.name, output_figures=output_figures)

    def _generate_data(self):

        data = []
        for repertoire in self.dataset.repertoires:
            data.append(self._compute_repertoire(repertoire))

        data = pd.concat(data)
        data = data.replace({"in": "In Frame",
                             "out": "Out of Frame",
                             "stop": "Stop Codon",
                             "total_reads": "Total Reads",
                             "unique_clonotypes": "Unique Clonotypes"})

        return data

    def _plot(self, data) -> ReportOutput:
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import STAP

        pandas2ri.activate()

        r_file_path = EnvironmentSettings.root_path / "source/visualization/SequencingDepthOverview.R"
        with r_file_path.open() as f:
            string = f.read()

        plot = STAP(string, "plot")

        PathBuilder.build(self.result_path)

        plot.plot_sequencing_depth_overview(data=data[[self.x, "value", "frame_type", "feature", "id"] + self.facets],
                                            x=self.x,
                                            color=self.color,
                                            facets=self.facets,
                                            palette=json.dumps(self.palette),
                                            nrow_distributions=self.nrow_distributions,
                                            nrow_scatterplot=self.nrow_scatterplot,
                                            height_distributions=self.height_distributions,
                                            height_scatterplot=self.height_scatterplot,
                                            width=self.width,
                                            result_path=str(self.result_path),
                                            result_name=self.result_name)

        return ReportOutput(path=self.result_path / f"{self.result_name}.pdf")

    def _compute_repertoire(self, repertoire):
        result = []
        for frame_type in SequenceFrameType:
            total_reads = self._compute_total_reads(repertoire, frame_type)
            unique_clonotypes = self._compute_unique_clonotypes(repertoire, frame_type)
            result.append({"value": total_reads, "frame_type": frame_type.name.lower(), "feature": "total_reads",
                           **repertoire.metadata, "id": repertoire.identifier})
            result.append(
                {"value": unique_clonotypes, "frame_type": frame_type.name.lower(), "feature": "unique_clonotypes",
                 **repertoire.metadata, "id": repertoire.identifier})
        return pd.DataFrame(result)

    def _compute_total_reads(self, repertoire: Repertoire, frame_type: SequenceFrameType):
        count = 0
        for sequence in repertoire.sequences:
            if sequence.metadata is not None and sequence.metadata.frame_type == frame_type:
                count += sequence.metadata.count
        return count

    def _compute_unique_clonotypes(self, repertoire: Repertoire, frame_type: SequenceFrameType):
        count = 0
        for sequence in repertoire.sequences:
            if sequence.metadata is not None and sequence.metadata.frame_type == frame_type:
                count += 1
        return count
