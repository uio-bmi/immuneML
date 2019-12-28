import json
from multiprocessing.pool import Pool

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.DataReport import DataReport
from source.util.PathBuilder import PathBuilder


class SequencingDepthOverview(DataReport):
    """
    Report to analyze sequencing depth as quantified by total number of reads per sample and total number of unique
    clonotypes per sample for in-frame, out-of-frame, and sequences containing stop codons.
    Distributions of each of these variables as well as the relationship between the variables are plotted in this
    report.
    @param x: discrete column name from metadata file with which to put on the x-axis and split samples by
    @param color: column name from metadata file to color samples by
    @param facets: metadata attributes to split points by in rows of the resulting plot-matrix
    @param palette: dictionary of desired color palette - if discrete, in the form value: color, if continuous, in the
    form colors: list of colors and breaks: list of values at which each color should be - for example:
    discrete: {"A": "blue", "B": "red", ...}
    continuous: {"colors": ["blue", "white", "red"], "breaks": [-1, 0, 1]}
    """

    def __init__(self, dataset: RepertoireDataset = None,
                 x: str = None,
                 color: str = None,
                 facets: list = [],
                 palette: dict = {},
                 nrow_distributions: int = 2,
                 nrow_scatterplot: int = 1,
                 height_distributions: float = 6.67,
                 height_scatterplot: float = 3.33,
                 width: float = 10,
                 result_name: str = "sequencing_depth_overview",
                 result_path: str = None,
                 batch_size: int = 1):

        DataReport.__init__(self, dataset=dataset, result_path=result_path)
        self.x = x
        self.color = color if color is not None else x
        self.facets = facets
        self.palette = palette
        self.nrow_distributions = nrow_distributions
        self.nrow_scatterplot = nrow_scatterplot
        self.height_distributions = height_distributions
        self.height_scatterplot = height_scatterplot
        self.width = width
        self.result_name = result_name
        self.batch_size = batch_size

    def generate(self):
        data = self.generate_data()
        self.plot(data)

    def generate_data(self):

        with Pool(self.batch_size, maxtasksperchild=1) as pool:
            data = pool.map(self._compute_repertoire, self.dataset.repertoires, chunksize=1)

        data = pd.concat(data)
        data = data.replace({"in": "In Frame",
                             "out": "Out of Frame",
                             "stop": "Stop Codon",
                             "total_reads": "Total Reads",
                             "unique_clonotypes": "Unique Clonotypes"})

        return data

    def plot(self, data):

        pandas2ri.activate()

        with open(
                EnvironmentSettings.root_path + "source/reports/data_reports/sequencing_depth_overview/SequencingDepthOverview.R") as f:
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
                                            result_path=self.result_path,
                                            result_name=self.result_name)

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

    def _compute_total_reads(self, repertoire: SequenceRepertoire, frame_type: SequenceFrameType):
        count = 0
        for sequence in repertoire.sequences:
            if sequence.metadata is not None and sequence.metadata.frame_type.upper() == frame_type.name:
                count += sequence.metadata.count
        return count

    def _compute_unique_clonotypes(self, repertoire: SequenceRepertoire, frame_type: SequenceFrameType):
        count = 0
        for sequence in repertoire.sequences:
            if sequence.metadata is not None and sequence.metadata.frame_type.upper() == frame_type.name:
                count += 1
        return count
