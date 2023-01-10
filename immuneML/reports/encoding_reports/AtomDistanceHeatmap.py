from pathlib import Path
import numpy as np
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
import plotly.express as px
from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder


class AtomDistanceHeatmap(EncodingReport):

    @classmethod
    def build_object(cls,
                     **kwargs):  # called when parsing YAML - all checks for parameters (if any) should be in this function
        return AtomDistanceHeatmap(**kwargs)

    def __init__(self, dataset: PDBDataset = None, result_path: Path = None,
                 name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)

    def check_prerequisites(
            self):  # called at runtime to check if the report can be run with params assigned at runtime (e.g., dataset is set at runtime)

        return True

    def _generate(self) -> ReportResult:  # the function that creates the report

        dataset = np.array(self.dataset.encoded_data.examples.copy(),dtype=float)
        output_files = []

        PathBuilder.build(self.result_path)

        for i in range(0, len(dataset)):

                file_name = Path(self.dataset.get_example_ids()[i])

                light_chain_distance = self.result_path / "{}_lightChainDistanceToAntigen.html".format(file_name)
                heavy_chain_distance = self.result_path / "{}_heavyChainDistanceToAntigen.html".format(file_name)

                light_chain_distance_to_antigen = self.remove_empty_rows(self.remove_infinite_values(dataset[i][0]))
                heavy_chain_distance_to_antigen = self.remove_empty_rows(self.remove_infinite_values(dataset[i][1]))

                light_chain_heatmap = px.imshow(light_chain_distance_to_antigen)
                light_chain_heatmap.write_html(str(light_chain_distance))

                heavy_chain_heatmap = px.imshow(heavy_chain_distance_to_antigen)
                heavy_chain_heatmap.write_html(str(heavy_chain_distance))


                output_files.append(ReportOutput(
                    name="{} - Heatmap of distance from carbon alphas in the LIGHT chain to carbon alphas in the antigen ".format(
                        file_name), path=light_chain_distance))
                output_files.append(ReportOutput(
                    name="{} - Heatmap of distance from carbon alphas in the HEAVY chain to carbon alphas in the antigen ".format(
                        file_name), path=heavy_chain_distance))


        return ReportResult(type(self).__name__, info="Heatmap", output_figures=output_files)


    def remove_infinite_values(self,arr):
        output = []
        for x in arr:
            output.append(x[np.isfinite(x)])

        return output

    def remove_empty_rows(self, arr):
        return np.array([row for row in arr if len(row)>0])