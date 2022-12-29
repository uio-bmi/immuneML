import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from plotly.subplots import make_subplots

import pandas as pd
import plotly.express as px
from Bio.PDB import PDBParser

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder



class PDBDataReport(EncodingReport):
    """
    Generates heatmaps of the distance between the light and heavy chain to the antigen in the given PDB files.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_pdb_heatmap: PDBDataReport

    """

    @classmethod
    def build_object(cls,
                     **kwargs):  # called when parsing YAML - all checks for parameters (if any) should be in this function
        return PDBDataReport(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, randomInt: int = 0):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.randomInt = randomInt
        self.batch_size = batch_size

    def check_prerequisites(
            self):  # called at runtime to check if the report can be run with params assigned at runtime (e.g., dataset is set at runtime)

        return True

    def _generate(self) -> ReportResult:  # the function that creates the report

        dataset = np.array(self.dataset.encoded_data.examples.copy(),dtype=float)
        output_files = []

        CDR = self.dataset.get_metadata(["CDR"])

        for i in range(0, len(dataset)):
                meta = str(CDR["CDR"][i])

                fileName = Path(self.dataset.get_example_ids()[i])
                PathBuilder.build(self.result_path)

                lightChainDistance = self.result_path / "{}_lightChainDistanceToAntigen.html".format(fileName)
                heavyChainDistance = self.result_path / "{}_heavyChainDistanceToAntigen.html".format(fileName)

                lightChainDistanceToAntigen = removeEmptyRows(removeInfValues(dataset[i][0]))
                heavyChainDistanceToAntigen = removeEmptyRows(removeInfValues(dataset[i][1]))

                figure = px.imshow(lightChainDistanceToAntigen)
                figure.write_html(str(lightChainDistance))

                fig2 = px.imshow(heavyChainDistanceToAntigen)
                fig2.write_html(str(heavyChainDistance))


                output_files.append(ReportOutput(
                    name="{} - Heatmap of distance from carbon alphas in the LIGHT chain to carbon alphas in the antigen ".format(
                        fileName), path=lightChainDistance))
                output_files.append(ReportOutput(
                    name="{} - Heatmap of distance from carbon alphas in the HEAVY chain to carbon alphas in the antigen ".format(
                        fileName), path=heavyChainDistance))


        return ReportResult(type(self).__name__, info="Heatmap", output_figures=output_files)


def removeInfValues(arr):
    output = []
    for x in arr:
        output.append(x[np.isfinite(x)])

    return output

def removeEmptyRows(arr):
    return np.array([row for row in arr if len(row)>0])